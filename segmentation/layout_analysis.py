import numpy as np
try:
    import cv2
except ImportError:
    print("OpenCV not available - using fallback implementations")
    cv2 = None

class LayoutAnalyzer:
    """
    Page layout analysis using recursive XY-cuts and connected component analysis
    """

    def __init__(self):
        self.min_text_height = 10
        self.min_text_width = 20
        self.white_space_threshold = 0.9  # Ratio of white pixels to consider as separator
        self.merge_threshold = 5  # Pixels - distance to merge nearby components

    def find_text_regions(self, binary_image):
        """Find text regions using connected component analysis"""
        if cv2 is not None:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8
            )

            text_regions = []
            for i in range(1, num_labels):  # Skip background
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                # Filter by size
                if w >= self.min_text_width and h >= self.min_text_height:
                    text_regions.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'centroid': centroids[i]
                    })

            return text_regions
        else:
            # Fallback implementation
            return self.simple_find_regions(binary_image)

    def simple_find_regions(self, binary_image):
        """Simple region finding without OpenCV"""
        h, w = binary_image.shape
        visited = np.zeros((h, w), dtype=bool)
        text_regions = []

        def flood_fill_bbox(start_y, start_x):
            """Find bounding box of connected component"""
            stack = [(start_y, start_x)]
            min_x, max_x = start_x, start_x
            min_y, max_y = start_y, start_y
            pixel_count = 0

            while stack:
                y, x = stack.pop()
                if (y < 0 or y >= h or x < 0 or x >= w or 
                    visited[y, x] or binary_image[y, x] == 255):
                    continue

                visited[y, x] = True
                pixel_count += 1

                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        stack.append((y + dy, x + dx))

            width = max_x - min_x + 1
            height = max_y - min_y + 1

            return (min_x, min_y, width, height), pixel_count

        # Find all connected components
        for y in range(h):
            for x in range(w):
                if not visited[y, x] and binary_image[y, x] == 0:  # Black pixel
                    bbox, area = flood_fill_bbox(y, x)

                    if (bbox[2] >= self.min_text_width and 
                        bbox[3] >= self.min_text_height):
                        text_regions.append({
                            'bbox': bbox,
                            'area': area,
                            'centroid': (bbox[0] + bbox[2]//2, bbox[1] + bbox[3]//2)
                        })

        return text_regions

    def merge_nearby_regions(self, regions):
        """Merge nearby text regions that likely belong together"""
        if not regions:
            return regions

        merged_regions = []
        used = set()

        for i, region1 in enumerate(regions):
            if i in used:
                continue

            x1, y1, w1, h1 = region1['bbox']
            merged_bbox = [x1, y1, x1 + w1, y1 + h1]  # [left, top, right, bottom]
            merged_area = region1['area']

            # Check for nearby regions to merge
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used:
                    continue

                x2, y2, w2, h2 = region2['bbox']

                # Check if regions are close enough to merge
                horizontal_gap = max(0, x2 - (x1 + w1), x1 - (x2 + w2))
                vertical_gap = max(0, y2 - (y1 + h1), y1 - (y2 + h2))

                if horizontal_gap <= self.merge_threshold and vertical_gap <= self.merge_threshold:
                    # Merge regions
                    merged_bbox[0] = min(merged_bbox[0], x2)  # left
                    merged_bbox[1] = min(merged_bbox[1], y2)  # top
                    merged_bbox[2] = max(merged_bbox[2], x2 + w2)  # right
                    merged_bbox[3] = max(merged_bbox[3], y2 + h2)  # bottom
                    merged_area += region2['area']
                    used.add(j)

            # Convert back to (x, y, w, h) format
            final_bbox = (
                merged_bbox[0], merged_bbox[1],
                merged_bbox[2] - merged_bbox[0],
                merged_bbox[3] - merged_bbox[1]
            )

            merged_regions.append({
                'bbox': final_bbox,
                'area': merged_area,
                'centroid': (final_bbox[0] + final_bbox[2]//2, 
                           final_bbox[1] + final_bbox[3]//2)
            })

        return merged_regions

    def xy_cut_segmentation(self, binary_image, bbox=None):
        """Recursive XY-cut algorithm for layout segmentation"""
        if bbox is None:
            bbox = (0, 0, binary_image.shape[1], binary_image.shape[0])

        x, y, w, h = bbox
        region = binary_image[y:y+h, x:x+w]

        # Base case: region too small
        if w < self.min_text_width * 2 or h < self.min_text_height * 2:
            return [bbox]

        # Try horizontal cut first
        horizontal_projection = np.mean(region == 255, axis=1)  # White pixel ratio

        # Find horizontal separator (high ratio of white pixels)
        cut_positions = []
        for i, ratio in enumerate(horizontal_projection):
            if ratio > self.white_space_threshold:
                cut_positions.append(i)

        if cut_positions:
            # Find the best cut position (center of white space)
            best_cut = self.find_best_cut_position(cut_positions, h)
            if best_cut is not None:
                # Recursively segment upper and lower parts
                upper_bbox = (x, y, w, best_cut)
                lower_bbox = (x, y + best_cut, w, h - best_cut)

                result = []
                result.extend(self.xy_cut_segmentation(binary_image, upper_bbox))
                result.extend(self.xy_cut_segmentation(binary_image, lower_bbox))
                return result

        # Try vertical cut
        vertical_projection = np.mean(region == 255, axis=0)  # White pixel ratio

        cut_positions = []
        for i, ratio in enumerate(vertical_projection):
            if ratio > self.white_space_threshold:
                cut_positions.append(i)

        if cut_positions:
            best_cut = self.find_best_cut_position(cut_positions, w)
            if best_cut is not None:
                # Recursively segment left and right parts
                left_bbox = (x, y, best_cut, h)
                right_bbox = (x + best_cut, y, w - best_cut, h)

                result = []
                result.extend(self.xy_cut_segmentation(binary_image, left_bbox))
                result.extend(self.xy_cut_segmentation(binary_image, right_bbox))
                return result

        # No good cut found, return current region
        return [bbox]

    def find_best_cut_position(self, cut_positions, dimension):
        """Find the best position to make a cut"""
        if not cut_positions:
            return None

        # Group consecutive positions
        groups = []
        current_group = [cut_positions[0]]

        for pos in cut_positions[1:]:
            if pos - current_group[-1] <= 2:  # Close positions
                current_group.append(pos)
            else:
                groups.append(current_group)
                current_group = [pos]
        groups.append(current_group)

        # Find the largest group (widest white space)
        if groups:
            largest_group = max(groups, key=len)
            center = int(np.mean(largest_group))

            # Ensure cut is not too close to edges
            min_margin = dimension // 10
            if center > min_margin and center < dimension - min_margin:
                return center

        return None

    def analyze_layout(self, binary_image):
        """Main layout analysis pipeline"""
        # Find individual text regions
        text_regions = self.find_text_regions(binary_image)

        # Merge nearby regions
        merged_regions = self.merge_nearby_regions(text_regions)

        # Apply XY-cut segmentation for hierarchical layout
        layout_regions = []
        for region in merged_regions:
            bbox = region['bbox']
            sub_regions = self.xy_cut_segmentation(binary_image, bbox)
            layout_regions.extend(sub_regions)

        # Sort regions by reading order (top to bottom, left to right)
        layout_regions.sort(key=lambda r: (r[1], r[0]))  # Sort by y, then x

        return {
            'text_regions': text_regions,
            'merged_regions': merged_regions,
            'layout_regions': layout_regions
        }

# Convenience function
def analyze_page_layout(binary_image):
    """Convenience function for layout analysis"""
    analyzer = LayoutAnalyzer()
    return analyzer.analyze_layout(binary_image)