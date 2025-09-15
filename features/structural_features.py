import numpy as np
try:
    import cv2
    from skimage import morphology, measure
except ImportError:
    print("OpenCV/scikit-image not available - using fallback implementations")
    cv2 = None
    morphology = None
    measure = None

class StructuralFeatureExtractor:
    """
    Extract structural features from character images including loops,
    endpoints, junctions, and stroke patterns
    """

    def __init__(self):
        self.target_size = (64, 64)

    def preprocess_character(self, char_image):
        """Preprocess character for structural analysis"""
        # Ensure binary image
        if len(char_image.shape) == 3:
            if cv2 is not None:
                char_image = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
            else:
                char_image = np.dot(char_image[...,:3], [0.2989, 0.5870, 0.1140])

        # Binarize if needed
        if char_image.max() > 1:
            binary = (char_image < 128).astype(np.uint8)  # Black = 1, White = 0
        else:
            binary = char_image.astype(np.uint8)

        # Resize to standard size
        binary_resized = self.simple_resize(binary, self.target_size)

        return binary_resized

    def simple_resize(self, image, target_size):
        """Simple image resizing"""
        h, w = image.shape
        target_h, target_w = target_size

        scale_h = h / target_h
        scale_w = w / target_w

        resized = np.zeros(target_size, dtype=image.dtype)

        for y in range(target_h):
            for x in range(target_w):
                orig_y = int(y * scale_h)
                orig_x = int(x * scale_w)
                orig_y = min(orig_y, h - 1)
                orig_x = min(orig_x, w - 1)
                resized[y, x] = image[orig_y, orig_x]

        return resized

    def skeletonize_image(self, binary_image):
        """Create skeleton of character for structural analysis"""
        if morphology is not None:
            skeleton = morphology.skeletonize(binary_image > 0)
            return skeleton.astype(np.uint8)
        else:
            return self.simple_skeletonize(binary_image)

    def simple_skeletonize(self, binary_image):
        """Simple skeletonization using erosion"""
        skeleton = np.copy(binary_image)

        # Iterative thinning
        for _ in range(10):  # Limited iterations for simplicity
            eroded = self.simple_erosion(skeleton)
            if np.array_equal(eroded, skeleton):
                break
            skeleton = eroded

        return skeleton

    def simple_erosion(self, binary_image):
        """Simple morphological erosion"""
        h, w = binary_image.shape
        eroded = np.zeros_like(binary_image)

        # 3x3 structuring element
        for y in range(1, h-1):
            for x in range(1, w-1):
                if binary_image[y, x] == 1:
                    # Check 8-neighborhood
                    neighborhood = binary_image[y-1:y+2, x-1:x+2]
                    if np.sum(neighborhood) >= 5:  # Keep if enough neighbors
                        eroded[y, x] = 1

        return eroded

    def count_endpoints(self, skeleton):
        """Count endpoints in skeleton"""
        if skeleton.sum() == 0:
            return 0

        h, w = skeleton.shape
        endpoints = 0

        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 1:
                    # Count neighbors in 8-connectivity
                    neighbors = skeleton[y-1:y+2, x-1:x+2]
                    neighbor_count = neighbors.sum() - 1  # Exclude center pixel

                    if neighbor_count == 1:  # Endpoint has only 1 neighbor
                        endpoints += 1

        return endpoints

    def count_junctions(self, skeleton):
        """Count junction points (intersections) in skeleton"""
        if skeleton.sum() == 0:
            return 0

        h, w = skeleton.shape
        junctions = 0

        for y in range(1, h-1):
            for x in range(1, w-1):
                if skeleton[y, x] == 1:
                    # Count neighbors in 8-connectivity
                    neighbors = skeleton[y-1:y+2, x-1:x+2]
                    neighbor_count = neighbors.sum() - 1  # Exclude center pixel

                    if neighbor_count >= 3:  # Junction has 3+ neighbors
                        junctions += 1

        return junctions

    def count_loops(self, binary_image):
        """Count closed loops in character"""
        if cv2 is not None:
            # Find contours
            contours, hierarchy = cv2.findContours(
                binary_image.astype(np.uint8), 
                cv2.RETR_TREE, 
                cv2.CHAIN_APPROX_SIMPLE
            )

            if hierarchy is not None:
                # Count internal contours (holes)
                loops = 0
                for i, h in enumerate(hierarchy[0]):
                    # h = [next, prev, first_child, parent]
                    if h[3] != -1:  # Has parent (internal contour)
                        loops += 1
                return loops
            else:
                return 0
        else:
            return self.simple_loop_detection(binary_image)

    def simple_loop_detection(self, binary_image):
        """Simple loop detection using flood fill"""
        # Invert image (background = 1, foreground = 0)
        inverted = 1 - binary_image
        visited = np.zeros_like(inverted, dtype=bool)
        loops = 0

        h, w = inverted.shape

        # Find all background regions
        for y in range(h):
            for x in range(w):
                if inverted[y, x] == 1 and not visited[y, x]:
                    # Check if this region is enclosed
                    region_pixels = self.flood_fill_region(inverted, visited, y, x)

                    # If region doesn't touch border, it's a loop
                    touches_border = any(
                        px == 0 or py == 0 or px == w-1 or py == h-1
                        for py, px in region_pixels
                    )

                    if not touches_border and len(region_pixels) > 5:
                        loops += 1

        return loops

    def flood_fill_region(self, image, visited, start_y, start_x):
        """Flood fill to find connected region"""
        h, w = image.shape
        stack = [(start_y, start_x)]
        region_pixels = []

        while stack:
            y, x = stack.pop()
            if (y < 0 or y >= h or x < 0 or x >= w or 
                visited[y, x] or image[y, x] == 0):
                continue

            visited[y, x] = True
            region_pixels.append((y, x))

            # Add 4-connected neighbors
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

        return region_pixels

    def compute_stroke_density(self, binary_image):
        """Compute stroke density in different regions"""
        h, w = binary_image.shape

        # Divide image into zones
        zones = {
            'top': binary_image[:h//3, :],
            'middle': binary_image[h//3:2*h//3, :],
            'bottom': binary_image[2*h//3:, :],
            'left': binary_image[:, :w//3],
            'center': binary_image[:, w//3:2*w//3],
            'right': binary_image[:, 2*w//3:]
        }

        densities = {}
        for zone_name, zone in zones.items():
            if zone.size > 0:
                densities[zone_name] = zone.sum() / zone.size
            else:
                densities[zone_name] = 0.0

        return densities

    def compute_aspect_ratios(self, binary_image):
        """Compute various aspect ratio features"""
        # Find bounding box of character
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)

        if not rows.any() or not cols.any():
            return {'aspect_ratio': 1.0, 'width_height_ratio': 1.0}

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        height = rmax - rmin + 1
        width = cmax - cmin + 1

        return {
            'aspect_ratio': width / height,
            'width_height_ratio': width / height,
            'bounding_box_ratio': (width * height) / binary_image.size
        }

    def compute_convex_hull_features(self, binary_image):
        """Compute convex hull related features"""
        if cv2 is None:
            return {'convexity': 1.0, 'solidity': 1.0}

        # Find contours
        contours, _ = cv2.findContours(
            binary_image.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return {'convexity': 1.0, 'solidity': 1.0}

        # Use largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute convex hull
        hull = cv2.convexHull(largest_contour)

        # Calculate features
        contour_area = cv2.contourArea(largest_contour)
        hull_area = cv2.contourArea(hull)

        if hull_area > 0:
            solidity = contour_area / hull_area
        else:
            solidity = 1.0

        # Convexity (ratio of perimeters)
        contour_perimeter = cv2.arcLength(largest_contour, True)
        hull_perimeter = cv2.arcLength(hull, True)

        if hull_perimeter > 0:
            convexity = hull_perimeter / contour_perimeter
        else:
            convexity = 1.0

        return {'convexity': convexity, 'solidity': solidity}

    def extract_structural_features(self, char_image):
        """Extract all structural features"""
        # Preprocess
        binary_char = self.preprocess_character(char_image)

        # Create skeleton
        skeleton = self.skeletonize_image(binary_char)

        # Extract features
        features = {}

        # Topological features
        features['endpoints'] = self.count_endpoints(skeleton)
        features['junctions'] = self.count_junctions(skeleton)
        features['loops'] = self.count_loops(binary_char)

        # Geometric features
        aspect_features = self.compute_aspect_ratios(binary_char)
        features.update(aspect_features)

        # Density features
        density_features = self.compute_stroke_density(binary_char)
        for zone, density in density_features.items():
            features[f'density_{zone}'] = density

        # Shape features
        hull_features = self.compute_convex_hull_features(binary_char)
        features.update(hull_features)

        # Additional features
        features['pixel_density'] = binary_char.sum() / binary_char.size
        features['skeleton_ratio'] = skeleton.sum() / max(binary_char.sum(), 1)

        # Convert to feature vector
        feature_vector = self.features_to_vector(features)

        return feature_vector

    def features_to_vector(self, features_dict):
        """Convert feature dictionary to vector"""
        # Define feature order for consistency
        feature_names = [
            'endpoints', 'junctions', 'loops',
            'aspect_ratio', 'width_height_ratio', 'bounding_box_ratio',
            'density_top', 'density_middle', 'density_bottom',
            'density_left', 'density_center', 'density_right',
            'convexity', 'solidity', 'pixel_density', 'skeleton_ratio'
        ]

        vector = []
        for name in feature_names:
            vector.append(features_dict.get(name, 0.0))

        return np.array(vector)

    def get_feature_dimension(self):
        """Get the dimension of feature vector"""
        # Create dummy image to get feature dimension
        dummy_image = np.zeros(self.target_size, dtype=np.uint8)
        features = self.extract_structural_features(dummy_image)
        return len(features)

# Convenience function
def extract_structural_features(char_image):
    """Convenience function for structural feature extraction"""
    extractor = StructuralFeatureExtractor()
    return extractor.extract_structural_features(char_image)