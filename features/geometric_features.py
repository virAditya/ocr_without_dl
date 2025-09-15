import numpy as np
try:
    import cv2
    from skimage import measure, transform
except ImportError:
    print("OpenCV/scikit-image not available - using fallback implementations")
    cv2 = None
    measure = None
    transform = None

class GeometricFeatureExtractor:
    """
    Extract geometric features from character images including Hu moments,
    centroid features, and geometric shape descriptors
    """

    def __init__(self):
        self.target_size = (64, 64)

    def preprocess_character(self, char_image):
        """Preprocess character for geometric analysis"""
        # Ensure grayscale
        if len(char_image.shape) == 3:
            if cv2 is not None:
                char_image = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
            else:
                char_image = np.dot(char_image[...,:3], [0.2989, 0.5870, 0.1140])

        # Resize to standard size
        resized = self.simple_resize(char_image, self.target_size)

        # Ensure proper intensity range
        if resized.max() > 1:
            # Convert to 0-1 range for moment calculations
            resized = resized.astype(float) / 255.0

        return resized

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

    def compute_moments(self, image):
        """Compute image moments"""
        if cv2 is not None:
            moments = cv2.moments(image)
            return moments
        else:
            return self.compute_moments_manual(image)

    def compute_moments_manual(self, image):
        """Manual computation of image moments"""
        h, w = image.shape
        moments = {}

        # Raw moments
        for p in range(4):  # Up to 3rd order
            for q in range(4 - p):
                moment = 0.0
                for y in range(h):
                    for x in range(w):
                        moment += (x ** p) * (y ** q) * image[y, x]
                moments[f'm{p}{q}'] = moment

        return moments

    def compute_central_moments(self, image, moments=None):
        """Compute central moments"""
        if moments is None:
            moments = self.compute_moments(image)

        # Centroid
        if moments.get('m00', 0) != 0:
            cx = moments.get('m10', 0) / moments['m00']
            cy = moments.get('m01', 0) / moments['m00']
        else:
            cx, cy = 0, 0

        h, w = image.shape
        central_moments = {}

        # Central moments
        for p in range(4):
            for q in range(4 - p):
                mu = 0.0
                for y in range(h):
                    for x in range(w):
                        mu += ((x - cx) ** p) * ((y - cy) ** q) * image[y, x]
                central_moments[f'mu{p}{q}'] = mu

        return central_moments, (cx, cy)

    def compute_hu_moments(self, image):
        """Compute Hu moments (7 rotation invariant moments)"""
        if cv2 is not None:
            moments = cv2.moments(image)
            hu_moments = cv2.HuMoments(moments).flatten()

            # Take log to reduce dynamic range
            hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

            return hu_moments
        else:
            return self.compute_hu_moments_manual(image)

    def compute_hu_moments_manual(self, image):
        """Manual computation of Hu moments"""
        moments = self.compute_moments(image)
        central_moments, _ = self.compute_central_moments(image, moments)

        # Normalized central moments
        m00 = central_moments.get('mu00', 1e-10)

        if m00 == 0:
            return np.zeros(7)

        # Calculate normalized central moments
        nu = {}
        for p in range(4):
            for q in range(4 - p):
                if p + q >= 2:
                    gamma = ((p + q) / 2) + 1
                    nu[f'nu{p}{q}'] = central_moments.get(f'mu{p}{q}', 0) / (m00 ** gamma)

        # Hu moment invariants
        hu1 = nu.get('nu20', 0) + nu.get('nu02', 0)

        hu2 = (nu.get('nu20', 0) - nu.get('nu02', 0))**2 + 4 * nu.get('nu11', 0)**2

        hu3 = (nu.get('nu30', 0) - 3*nu.get('nu12', 0))**2 + (3*nu.get('nu21', 0) - nu.get('nu03', 0))**2

        hu4 = (nu.get('nu30', 0) + nu.get('nu12', 0))**2 + (nu.get('nu21', 0) + nu.get('nu03', 0))**2

        hu5 = ((nu.get('nu30', 0) - 3*nu.get('nu12', 0)) * (nu.get('nu30', 0) + nu.get('nu12', 0)) * 
               ((nu.get('nu30', 0) + nu.get('nu12', 0))**2 - 3*(nu.get('nu21', 0) + nu.get('nu03', 0))**2) +
               (3*nu.get('nu21', 0) - nu.get('nu03', 0)) * (nu.get('nu21', 0) + nu.get('nu03', 0)) * 
               (3*(nu.get('nu30', 0) + nu.get('nu12', 0))**2 - (nu.get('nu21', 0) + nu.get('nu03', 0))**2))

        hu6 = ((nu.get('nu20', 0) - nu.get('nu02', 0)) * 
               ((nu.get('nu30', 0) + nu.get('nu12', 0))**2 - (nu.get('nu21', 0) + nu.get('nu03', 0))**2) +
               4*nu.get('nu11', 0) * (nu.get('nu30', 0) + nu.get('nu12', 0)) * (nu.get('nu21', 0) + nu.get('nu03', 0)))

        hu7 = ((3*nu.get('nu21', 0) - nu.get('nu03', 0)) * (nu.get('nu30', 0) + nu.get('nu12', 0)) *
               ((nu.get('nu30', 0) + nu.get('nu12', 0))**2 - 3*(nu.get('nu21', 0) + nu.get('nu03', 0))**2) -
               (nu.get('nu30', 0) - 3*nu.get('nu12', 0)) * (nu.get('nu21', 0) + nu.get('nu03', 0)) *
               (3*(nu.get('nu30', 0) + nu.get('nu12', 0))**2 - (nu.get('nu21', 0) + nu.get('nu03', 0))**2))

        hu_moments = np.array([hu1, hu2, hu3, hu4, hu5, hu6, hu7])

        # Take log to reduce dynamic range
        hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

        return hu_moments

    def compute_centroid_features(self, image):
        """Compute centroid-based features"""
        moments = self.compute_moments(image)

        if moments.get('m00', 0) != 0:
            cx = moments.get('m10', 0) / moments['m00']
            cy = moments.get('m01', 0) / moments['m00']
        else:
            cx, cy = image.shape[1] / 2, image.shape[0] / 2

        h, w = image.shape

        # Normalize centroid to [0, 1]
        norm_cx = cx / w
        norm_cy = cy / h

        # Distance from center
        center_x, center_y = w / 2, h / 2
        dist_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        norm_dist_from_center = dist_from_center / np.sqrt(center_x**2 + center_y**2)

        return {
            'centroid_x': norm_cx,
            'centroid_y': norm_cy,
            'dist_from_center': norm_dist_from_center
        }

    def compute_contour_features(self, image):
        """Compute contour-based geometric features"""
        # Convert to binary for contour detection
        if image.max() <= 1:
            binary = (image > 0.5).astype(np.uint8)
        else:
            binary = (image < 128).astype(np.uint8)

        if cv2 is not None:
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return self.default_contour_features()

            # Use largest contour
            largest_contour = max(contours, key=cv2.contourArea)

            # Area and perimeter
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            # Compactness (circularity)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
            else:
                compactness = 0

            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            extent = area / (w * h) if w * h > 0 else 0

            # Minimum enclosing circle
            (circle_x, circle_y), radius = cv2.minEnclosingCircle(largest_contour)
            circle_area = np.pi * radius ** 2
            circularity = area / circle_area if circle_area > 0 else 0

            return {
                'area': area / (image.shape[0] * image.shape[1]),  # Normalized
                'perimeter': perimeter / (2 * (image.shape[0] + image.shape[1])),  # Normalized
                'compactness': compactness,
                'extent': extent,
                'circularity': circularity
            }
        else:
            return self.simple_contour_features(binary)

    def default_contour_features(self):
        """Default contour features when no contours found"""
        return {
            'area': 0.0,
            'perimeter': 0.0,
            'compactness': 0.0,
            'extent': 0.0,
            'circularity': 0.0
        }

    def simple_contour_features(self, binary_image):
        """Simple contour feature computation without OpenCV"""
        # Calculate area (number of pixels)
        area = np.sum(binary_image)
        total_pixels = binary_image.shape[0] * binary_image.shape[1]

        if area == 0:
            return self.default_contour_features()

        # Find bounding box
        rows = np.any(binary_image, axis=1)
        cols = np.any(binary_image, axis=0)

        if not rows.any() or not cols.any():
            return self.default_contour_features()

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        bbox_area = (rmax - rmin + 1) * (cmax - cmin + 1)
        extent = area / bbox_area if bbox_area > 0 else 0

        # Approximate perimeter using edge detection
        perimeter = self.estimate_perimeter(binary_image)

        # Compactness
        if perimeter > 0:
            compactness = 4 * np.pi * area / (perimeter ** 2)
        else:
            compactness = 0

        return {
            'area': area / total_pixels,
            'perimeter': perimeter / (2 * sum(binary_image.shape)),
            'compactness': compactness,
            'extent': extent,
            'circularity': compactness  # Approximation
        }

    def estimate_perimeter(self, binary_image):
        """Estimate perimeter by counting edge pixels"""
        h, w = binary_image.shape
        perimeter = 0

        for y in range(h):
            for x in range(w):
                if binary_image[y, x] == 1:
                    # Check if it's an edge pixel (has at least one background neighbor)
                    is_edge = False
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if (ny < 0 or ny >= h or nx < 0 or nx >= w or 
                                binary_image[ny, nx] == 0):
                                is_edge = True
                                break
                        if is_edge:
                            break

                    if is_edge:
                        perimeter += 1

        return perimeter

    def compute_orientation_features(self, image):
        """Compute orientation-based features"""
        central_moments, centroid = self.compute_central_moments(image)

        mu20 = central_moments.get('mu20', 0)
        mu02 = central_moments.get('mu02', 0)
        mu11 = central_moments.get('mu11', 0)

        # Principal axis orientation
        if mu20 != mu02:
            orientation = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
        else:
            orientation = 0

        # Major and minor axis lengths
        mu20_norm = mu20 / central_moments.get('mu00', 1)
        mu02_norm = mu02 / central_moments.get('mu00', 1)
        mu11_norm = mu11 / central_moments.get('mu00', 1)

        eigenvals = [
            mu20_norm + mu02_norm + np.sqrt(4 * mu11_norm**2 + (mu20_norm - mu02_norm)**2),
            mu20_norm + mu02_norm - np.sqrt(4 * mu11_norm**2 + (mu20_norm - mu02_norm)**2)
        ]

        major_axis = 2 * np.sqrt(max(eigenvals, key=abs))
        minor_axis = 2 * np.sqrt(min(eigenvals, key=abs))

        eccentricity = 0 if minor_axis == 0 else np.sqrt(1 - (minor_axis / major_axis)**2)

        return {
            'orientation': orientation,
            'major_axis': major_axis,
            'minor_axis': minor_axis,
            'eccentricity': eccentricity
        }

    def extract_geometric_features(self, char_image):
        """Extract all geometric features"""
        # Preprocess
        processed_image = self.preprocess_character(char_image)

        # Extract different types of geometric features
        features = {}

        # Hu moments (7 features)
        hu_moments = self.compute_hu_moments(processed_image)
        for i, hu in enumerate(hu_moments):
            features[f'hu_moment_{i+1}'] = hu

        # Centroid features
        centroid_features = self.compute_centroid_features(processed_image)
        features.update(centroid_features)

        # Contour features
        contour_features = self.compute_contour_features(processed_image)
        features.update(contour_features)

        # Orientation features
        orientation_features = self.compute_orientation_features(processed_image)
        features.update(orientation_features)

        # Convert to feature vector
        feature_vector = self.features_to_vector(features)

        return feature_vector

    def features_to_vector(self, features_dict):
        """Convert feature dictionary to vector"""
        # Define feature order for consistency
        feature_names = [
            'hu_moment_1', 'hu_moment_2', 'hu_moment_3', 'hu_moment_4',
            'hu_moment_5', 'hu_moment_6', 'hu_moment_7',
            'centroid_x', 'centroid_y', 'dist_from_center',
            'area', 'perimeter', 'compactness', 'extent', 'circularity',
            'orientation', 'major_axis', 'minor_axis', 'eccentricity'
        ]

        vector = []
        for name in feature_names:
            vector.append(features_dict.get(name, 0.0))

        return np.array(vector)

    def get_feature_dimension(self):
        """Get the dimension of feature vector"""
        # Create dummy image to get feature dimension
        dummy_image = np.zeros(self.target_size, dtype=np.uint8)
        features = self.extract_geometric_features(dummy_image)
        return len(features)

# Convenience function
def extract_geometric_features(char_image):
    """Convenience function for geometric feature extraction"""
    extractor = GeometricFeatureExtractor()
    return extractor.extract_geometric_features(char_image)