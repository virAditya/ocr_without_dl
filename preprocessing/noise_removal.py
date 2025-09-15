import numpy as np
try:
    import cv2
except ImportError:
    print("OpenCV not available - using fallback implementations")
    cv2 = None

class NoiseRemover:
    """
    Noise removal using connected component analysis and morphological operations
    """

    def __init__(self):
        self.min_component_size = 10  # Minimum pixels for valid component
        self.max_component_size = 10000  # Maximum pixels for valid component
        self.aspect_ratio_threshold = 0.1  # Minimum width/height ratio

    def connected_components_filter(self, binary_image):
        """Remove noise using connected component analysis"""
        if cv2 is not None:
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8
            )

            # Create filtered image
            filtered = np.zeros_like(binary_image)

            for i in range(1, num_labels):  # Skip background (label 0)
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]

                # Filter by size
                if area < self.min_component_size or area > self.max_component_size:
                    continue

                # Filter by aspect ratio (remove very thin lines that might be noise)
                aspect_ratio = min(width, height) / max(width, height)
                if aspect_ratio < self.aspect_ratio_threshold:
                    continue

                # Keep this component
                filtered[labels == i] = 255

            return filtered
        else:
            # Fallback implementation using simple connected components
            return self.simple_connected_components(binary_image)

    def simple_connected_components(self, binary_image):
        """Simple connected component implementation without OpenCV"""
        h, w = binary_image.shape
        visited = np.zeros((h, w), dtype=bool)
        filtered = np.zeros_like(binary_image)

        def flood_fill(start_y, start_x):
            """Flood fill algorithm to find connected component"""
            stack = [(start_y, start_x)]
            component = []

            while stack:
                y, x = stack.pop()
                if (y < 0 or y >= h or x < 0 or x >= w or 
                    visited[y, x] or binary_image[y, x] == 255):
                    continue

                visited[y, x] = True
                component.append((y, x))

                # Check 8-connected neighbors
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        stack.append((y + dy, x + dx))

            return component

        # Find all connected components
        for y in range(h):
            for x in range(w):
                if not visited[y, x] and binary_image[y, x] == 0:  # Black pixel
                    component = flood_fill(y, x)

                    if len(component) >= self.min_component_size:
                        # Calculate bounding box
                        ys = [p[0] for p in component]
                        xs = [p[1] for p in component]
                        width = max(xs) - min(xs) + 1
                        height = max(ys) - min(ys) + 1

                        # Check aspect ratio
                        aspect_ratio = min(width, height) / max(width, height)
                        if aspect_ratio >= self.aspect_ratio_threshold:
                            # Keep this component
                            for py, px in component:
                                filtered[py, px] = 0  # Black for text

        # Invert to match expected output (white background, black text)
        return 255 - filtered

    def morphological_noise_removal(self, binary_image):
        """Remove noise using morphological operations"""
        if cv2 is not None:
            # Opening to remove small noise
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            opened = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel_open)

            # Closing to fill small gaps
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close)

            return closed
        else:
            # Simple erosion-dilation fallback
            return self.simple_morphology(binary_image)

    def simple_morphology(self, binary_image):
        """Simple morphological operations without OpenCV"""
        h, w = binary_image.shape

        # Simple erosion (remove single pixel noise)
        eroded = np.copy(binary_image)
        for y in range(1, h-1):
            for x in range(1, w-1):
                if binary_image[y, x] == 0:  # Black pixel
                    # Check if isolated
                    neighbors = binary_image[y-1:y+2, x-1:x+2]
                    if np.sum(neighbors == 0) <= 2:  # Less than 2 black neighbors
                        eroded[y, x] = 255  # Remove it

        # Simple dilation (restore important structures)
        dilated = np.copy(eroded)
        for y in range(1, h-1):
            for x in range(1, w-1):
                if eroded[y, x] == 255:  # White pixel
                    neighbors = eroded[y-1:y+2, x-1:x+2]
                    if np.sum(neighbors == 0) >= 3:  # Has 3+ black neighbors
                        dilated[y, x] = 0  # Fill gap

        return dilated

    def remove_border_noise(self, binary_image, border_size=5):
        """Remove noise touching image borders"""
        h, w = binary_image.shape
        cleaned = np.copy(binary_image)

        # Set border regions to white (background)
        cleaned[:border_size, :] = 255  # Top
        cleaned[-border_size:, :] = 255  # Bottom  
        cleaned[:, :border_size] = 255  # Left
        cleaned[:, -border_size:] = 255  # Right

        return cleaned

    def filter_by_position(self, binary_image):
        """Filter components based on position (remove margin noise)"""
        h, w = binary_image.shape
        margin_ratio = 0.05  # 5% margin

        margin_h = int(h * margin_ratio)
        margin_w = int(w * margin_ratio)

        if cv2 is not None:
            # Find connected components
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
                binary_image, connectivity=8
            )

            filtered = np.copy(binary_image)

            for i in range(1, num_labels):
                left = stats[i, cv2.CC_STAT_LEFT]
                top = stats[i, cv2.CC_STAT_TOP]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]

                # Check if component is in margin area
                if (left < margin_w or top < margin_h or 
                    left + width > w - margin_w or 
                    top + height > h - margin_h):
                    # Remove this component
                    filtered[labels == i] = 255

            return filtered
        else:
            # Simple margin cleaning
            return self.remove_border_noise(binary_image, max(margin_h, margin_w))

    def clean_noise(self, binary_image):
        """Main noise removal pipeline"""
        # Apply morphological noise removal
        cleaned = self.morphological_noise_removal(binary_image)

        # Remove components based on size and shape
        cleaned = self.connected_components_filter(cleaned)

        # Remove border and margin noise
        cleaned = self.filter_by_position(cleaned)

        return cleaned

# Convenience function
def remove_noise(binary_image):
    """Convenience function for noise removal"""
    noise_remover = NoiseRemover()
    return noise_remover.clean_noise(binary_image)