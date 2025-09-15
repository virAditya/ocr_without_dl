import numpy as np
try:
    import cv2
    from scipy import ndimage
except ImportError:
    print("OpenCV/SciPy not available - using fallback implementations")
    cv2 = None
    ndimage = None

class CharacterSegmenter:
    """
    Character segmentation using multiple methods: vertical projection, 
    watershed segmentation, and contour analysis
    """

    def __init__(self):
        self.min_char_width = 5
        self.max_char_width = 100
        self.min_char_height = 8
        self.char_spacing_threshold = 0.2  # Minimum space between characters

    def vertical_projection(self, line_image):
        """Calculate vertical projection profile"""
        projection = np.sum(line_image == 0, axis=0)  # Count black pixels
        return projection

    def find_character_boundaries_projection(self, line_image):
        """Find character boundaries using vertical projection"""
        projection = self.vertical_projection(line_image)
        height, width = line_image.shape

        # Smooth projection
        if width > 5:
            kernel_size = min(3, width // 10)
            if kernel_size >= 1:
                kernel = np.ones(kernel_size) / kernel_size
                projection = np.convolve(projection, kernel, mode='same')

        # Find character segments
        boundaries = []
        in_character = False
        char_start = 0
        min_gap = max(1, int(height * self.char_spacing_threshold))

        for i, value in enumerate(projection):
            if not in_character and value > 0:
                # Start of character
                in_character = True
                char_start = i
            elif in_character and value == 0:
                # Potential end of character, check for minimum gap
                gap_length = 0
                j = i
                while j < len(projection) and projection[j] == 0:
                    gap_length += 1
                    j += 1

                if gap_length >= min_gap or j == len(projection):
                    # End of character
                    char_width = i - char_start
                    if (char_width >= self.min_char_width and 
                        char_width <= self.max_char_width):
                        boundaries.append((char_start, i))
                    in_character = False

        # Handle case where character extends to end
        if in_character:
            char_width = width - char_start
            if (char_width >= self.min_char_width and 
                char_width <= self.max_char_width):
                boundaries.append((char_start, width))

        return boundaries

    def watershed_segmentation(self, line_image):
        """Use watershed algorithm for touching character separation"""
        if cv2 is None or ndimage is None:
            return self.fallback_touching_separation(line_image)

        # Distance transform
        dist_transform = cv2.distanceTransform(line_image, cv2.DIST_L2, 5)

        # Find local maxima (character centers)
        local_maxima = ndimage.maximum_filter(dist_transform, size=5) == dist_transform
        local_maxima = local_maxima & (dist_transform > np.max(dist_transform) * 0.3)

        # Create markers for watershed
        markers, num_markers = ndimage.label(local_maxima)

        if num_markers > 1:
            # Apply watershed
            watershed_result = cv2.watershed(
                cv2.cvtColor(line_image, cv2.COLOR_GRAY2BGR), 
                markers
            )

            # Extract boundaries
            boundaries = []
            for i in range(1, num_markers + 1):
                mask = (watershed_result == i).astype(np.uint8) * 255
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Get bounding box of largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    if (w >= self.min_char_width and w <= self.max_char_width and
                        h >= self.min_char_height):
                        boundaries.append((x, x + w))

            return sorted(boundaries)
        else:
            return []

    def fallback_touching_separation(self, line_image):
        """Fallback method for separating touching characters"""
        height, width = line_image.shape
        projection = self.vertical_projection(line_image)

        # Find local minima in projection (potential separation points)
        minima = []
        for i in range(1, len(projection) - 1):
            if (projection[i] < projection[i-1] and 
                projection[i] < projection[i+1] and
                projection[i] < np.mean(projection) * 0.5):
                minima.append(i)

        # Convert minima to character boundaries
        if not minima:
            return [(0, width)]

        boundaries = []
        start = 0

        for minimum in minima:
            if minimum - start >= self.min_char_width:
                boundaries.append((start, minimum))
                start = minimum

        # Add final segment
        if width - start >= self.min_char_width:
            boundaries.append((start, width))

        return boundaries

    def contour_based_segmentation(self, line_image):
        """Character segmentation using contour analysis"""
        if cv2 is None:
            return self.find_character_boundaries_projection(line_image)

        # Find contours
        contours, _ = cv2.findContours(
            255 - line_image,  # Invert for contour detection
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract bounding boxes
        boundaries = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter by size
            if (w >= self.min_char_width and w <= self.max_char_width and
                h >= self.min_char_height):
                boundaries.append((x, x + w))

        # Sort by x-coordinate
        boundaries.sort()

        # Merge overlapping boundaries
        merged_boundaries = self.merge_overlapping_boundaries(boundaries)

        return merged_boundaries

    def merge_overlapping_boundaries(self, boundaries):
        """Merge overlapping character boundaries"""
        if not boundaries:
            return []

        merged = []
        current_start, current_end = boundaries[0]

        for start, end in boundaries[1:]:
            if start <= current_end + 2:  # Allow small gap
                # Merge boundaries
                current_end = max(current_end, end)
            else:
                # Add current boundary and start new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end

        # Add final boundary
        merged.append((current_start, current_end))

        return merged

    def validate_character_segments(self, line_image, boundaries):
        """Validate character segments using statistical analysis"""
        if not boundaries:
            return []

        height = line_image.shape[0]
        valid_segments = []

        # Calculate statistics for size validation
        widths = [end - start for start, end in boundaries]
        if widths:
            median_width = np.median(widths)
            width_std = np.std(widths)

            for start, end in boundaries:
                char_width = end - start
                char_region = line_image[:, start:end]

                # Size validation
                width_z_score = abs(char_width - median_width) / max(width_std, 1)
                if width_z_score > 3:  # Outlier detection
                    continue

                # Content validation
                black_pixel_ratio = np.sum(char_region == 0) / char_region.size
                if black_pixel_ratio < 0.05:  # Less than 5% content
                    continue

                # Aspect ratio validation
                aspect_ratio = char_width / height
                if aspect_ratio < 0.1 or aspect_ratio > 2.0:  # Reasonable aspect ratios
                    continue

                valid_segments.append((start, end))

        return valid_segments

    def resolve_conflicts(self, projection_boundaries, contour_boundaries):
        """Resolve conflicts between different segmentation methods"""
        if not projection_boundaries and not contour_boundaries:
            return []

        if not projection_boundaries:
            return contour_boundaries

        if not contour_boundaries:
            return projection_boundaries

        # Use contour boundaries as primary, fill gaps with projection
        final_boundaries = []

        # Start with contour boundaries
        for contour_bound in contour_boundaries:
            # Find overlapping projection boundaries
            overlapping = []
            for proj_bound in projection_boundaries:
                if (proj_bound[1] > contour_bound[0] and 
                    proj_bound[0] < contour_bound[1]):
                    overlapping.append(proj_bound)

            if overlapping:
                # Use the boundary with better coverage
                best_bound = max(overlapping, key=lambda x: x[1] - x[0])
                final_boundaries.append(best_bound)
            else:
                final_boundaries.append(contour_bound)

        # Fill gaps with projection boundaries
        if final_boundaries:
            for proj_bound in projection_boundaries:
                is_covered = any(
                    proj_bound[0] >= fb[0] and proj_bound[1] <= fb[1]
                    for fb in final_boundaries
                )
                if not is_covered:
                    final_boundaries.append(proj_bound)

        # Sort and merge
        final_boundaries.sort()
        return self.merge_overlapping_boundaries(final_boundaries)

    def segment_characters(self, line_image):
        """Main character segmentation pipeline"""
        height, width = line_image.shape

        # Method 1: Vertical projection
        projection_boundaries = self.find_character_boundaries_projection(line_image)

        # Method 2: Contour analysis
        contour_boundaries = self.contour_based_segmentation(line_image)

        # Method 3: Watershed (for touching characters)
        watershed_boundaries = self.watershed_segmentation(line_image)

        # Resolve conflicts between methods
        primary_boundaries = self.resolve_conflicts(projection_boundaries, contour_boundaries)

        # Add watershed results for touching characters
        if watershed_boundaries:
            all_boundaries = primary_boundaries + watershed_boundaries
            all_boundaries = self.merge_overlapping_boundaries(sorted(all_boundaries))
        else:
            all_boundaries = primary_boundaries

        # Validate segments
        valid_segments = self.validate_character_segments(line_image, all_boundaries)

        # Create character regions
        character_regions = []
        for start, end in valid_segments:
            char_image = line_image[:, start:end]
            character_regions.append({
                'image': char_image,
                'bbox': (start, 0, end - start, height),
                'width': end - start,
                'height': height
            })

        return character_regions

# Convenience function
def segment_characters(line_image):
    """Convenience function for character segmentation"""
    segmenter = CharacterSegmenter()
    return segmenter.segment_characters(line_image)