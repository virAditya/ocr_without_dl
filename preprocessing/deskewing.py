import numpy as np
try:
    import cv2
except ImportError:
    print("OpenCV not available - using fallback implementations")
    cv2 = None

class ImageDeskewer:
    """
    Deskewing implementation using Hough transform and projection analysis
    """

    def __init__(self):
        self.angle_range = 45  # degrees
        self.angle_precision = 0.5  # degrees

    def detect_skew_hough(self, binary_image):
        """Detect skew angle using Hough line transform"""
        if cv2 is not None:
            # Apply Hough line transform
            lines = cv2.HoughLines(binary_image, 1, np.pi/180, threshold=100)

            if lines is not None:
                angles = []
                for line in lines:
                    rho, theta = line[0]
                    angle = np.degrees(theta) - 90
                    # Filter angles within reasonable range
                    if abs(angle) < self.angle_range:
                        angles.append(angle)

                if angles:
                    # Return median angle for robustness
                    return np.median(angles)

            return 0.0
        else:
            # Fallback: use projection method
            return self.detect_skew_projection(binary_image)

    def detect_skew_projection(self, binary_image):
        """Detect skew using projection profile analysis"""
        h, w = binary_image.shape
        angles = np.arange(-self.angle_range, self.angle_range, self.angle_precision)
        max_variance = 0
        best_angle = 0

        for angle in angles:
            # Rotate image
            if cv2 is not None:
                center = (w//2, h//2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(binary_image, rotation_matrix, (w, h))
            else:
                # Simple rotation fallback
                rotated = self.simple_rotate(binary_image, angle)

            # Calculate horizontal projection
            horizontal_proj = np.sum(rotated == 0, axis=1)  # Count black pixels

            # Calculate variance of projection
            variance = np.var(horizontal_proj)

            if variance > max_variance:
                max_variance = variance
                best_angle = angle

        return best_angle

    def simple_rotate(self, image, angle_degrees):
        """Simple rotation fallback without OpenCV"""
        angle_rad = np.radians(angle_degrees)
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)

        h, w = image.shape
        # Simple rotation - not perfect but functional
        center_x, center_y = w // 2, h // 2
        rotated = np.zeros_like(image)

        for y in range(h):
            for x in range(w):
                # Rotate coordinates
                x_centered = x - center_x
                y_centered = y - center_y

                new_x = int(x_centered * cos_angle - y_centered * sin_angle + center_x)
                new_y = int(x_centered * sin_angle + y_centered * cos_angle + center_y)

                if 0 <= new_x < w and 0 <= new_y < h:
                    rotated[y, x] = image[new_y, new_x]

        return rotated

    def correct_skew(self, binary_image, angle):
        """Correct skew by rotating image"""
        if abs(angle) < 0.1:  # No correction needed
            return binary_image

        h, w = binary_image.shape

        if cv2 is not None:
            center = (w//2, h//2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Calculate new image dimensions to avoid cropping
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_w = int((h * sin_angle) + (w * cos_angle))
            new_h = int((h * cos_angle) + (w * sin_angle))

            # Adjust rotation matrix for new center
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]

            corrected = cv2.warpAffine(binary_image, rotation_matrix, (new_w, new_h), 
                                     flags=cv2.INTER_CUBIC, borderValue=255)
            return corrected
        else:
            # Use simple rotation fallback
            return self.simple_rotate(binary_image, angle)

    def validate_correction(self, original, corrected):
        """Validate deskewing quality"""
        # Calculate horizontal projection variance
        orig_proj = np.sum(original == 0, axis=1)
        corr_proj = np.sum(corrected == 0, axis=1)

        orig_variance = np.var(orig_proj)
        corr_variance = np.var(corr_proj)

        # Higher variance in horizontal projection indicates better deskewing
        improvement_ratio = corr_variance / (orig_variance + 1e-8)

        return {
            'improvement_ratio': improvement_ratio,
            'is_improved': improvement_ratio > 1.1,
            'original_variance': orig_variance,
            'corrected_variance': corr_variance
        }

    def deskew(self, binary_image):
        """Main deskewing pipeline"""
        # Detect skew angle
        angle = self.detect_skew_hough(binary_image)

        # Correct skew
        corrected = self.correct_skew(binary_image, angle)

        # Validate correction
        validation = self.validate_correction(binary_image, corrected)

        return {
            'corrected_image': corrected,
            'detected_angle': angle,
            'validation': validation
        }

# Convenience function
def deskew_image(binary_image):
    """Convenience function for image deskewing"""
    deskewer = ImageDeskewer()
    result = deskewer.deskew(binary_image)
    return result['corrected_image']