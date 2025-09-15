import numpy as np
try:
    import cv2
except ImportError:
    print("OpenCV not available - using fallback implementations")
    cv2 = None

class ImageBinarizer:
    """
    Enhanced binarization with edge priors and morphological refinements
    """

    def __init__(self):
        self.block_size = 11
        self.c_constant = 2

    def otsu_threshold(self, image):
        """Otsu's automatic threshold selection"""
        if cv2 is not None:
            _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return binary
        else:
            # Fallback implementation
            hist, bins = np.histogram(image.ravel(), 256, [0, 256])
            total_pixels = image.size
            sum_total = 0
            for i in range(256):
                sum_total += i * hist[i]

            sum_background = 0
            weight_background = 0
            max_variance = 0
            threshold = 0

            for i in range(256):
                weight_background += hist[i]
                if weight_background == 0:
                    continue

                weight_foreground = total_pixels - weight_background
                if weight_foreground == 0:
                    break

                sum_background += i * hist[i]
                mean_background = sum_background / weight_background
                mean_foreground = (sum_total - sum_background) / weight_foreground

                variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2

                if variance > max_variance:
                    max_variance = variance
                    threshold = i

            binary = np.where(image > threshold, 255, 0).astype(np.uint8)
            return binary

    def adaptive_threshold(self, image):
        """Adaptive thresholding for varying lighting conditions"""
        if cv2 is not None:
            return cv2.adaptiveThreshold(
                image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, self.block_size, self.c_constant
            )
        else:
            # Simple fallback - divide image into blocks and apply local thresholding
            h, w = image.shape
            binary = np.zeros_like(image)
            block_h, block_w = h // 10, w // 10

            for i in range(0, h, block_h):
                for j in range(0, w, block_w):
                    block = image[i:i+block_h, j:j+block_w]
                    if block.size > 0:
                        threshold = np.mean(block) - self.c_constant
                        binary[i:i+block_h, j:j+block_w] = np.where(block > threshold, 255, 0)

            return binary.astype(np.uint8)

    def edge_guided_binarization(self, image):
        """Enhanced binarization using edge priors"""
        if cv2 is not None:
            # Detect edges using Canny
            edges = cv2.Canny(image, 50, 150)

            # Apply morphological operations to strengthen edges
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            edges_dilated = cv2.dilate(edges, kernel, iterations=1)

            # Combine adaptive threshold with edge information
            adaptive_binary = self.adaptive_threshold(image)

            # Use edges to guide binarization
            result = cv2.bitwise_or(adaptive_binary, edges_dilated)

            return result
        else:
            # Fallback - just use adaptive threshold
            return self.adaptive_threshold(image)

    def morphological_cleanup(self, binary_image):
        """Apply morphological operations for cleanup"""
        if cv2 is not None:
            # Remove noise
            kernel_noise = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_noise)

            # Close gaps in characters
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)

            return cleaned
        else:
            # Simple fallback - basic noise removal
            return binary_image

    def binarize(self, image_path):
        """Main binarization pipeline"""
        if cv2 is not None:
            # Read image
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Could not read image: {image_path}")
        else:
            # Fallback image reading
            from PIL import Image
            pil_image = Image.open(image_path).convert('L')
            image = np.array(pil_image)

        # Apply edge-guided binarization
        binary = self.edge_guided_binarization(image)

        # Apply morphological cleanup
        cleaned = self.morphological_cleanup(binary)

        return cleaned

# Example usage function
def binarize_image(image_path, output_path=None):
    """Convenience function for image binarization"""
    binarizer = ImageBinarizer()
    result = binarizer.binarize(image_path)

    if output_path and cv2 is not None:
        cv2.imwrite(output_path, result)

    return result