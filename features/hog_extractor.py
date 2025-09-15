import numpy as np
try:
    from skimage.feature import hog
    from skimage import transform
    import cv2
except ImportError:
    print("scikit-image/OpenCV not available - using fallback implementations")
    hog = None
    transform = None
    cv2 = None

class HOGExtractor:
    """
    Histogram of Oriented Gradients (HOG) feature extractor for character recognition
    """

    def __init__(self):
        # Standard HOG parameters
        self.orientations = 9
        self.pixels_per_cell = (8, 8)
        self.cells_per_block = (2, 2)
        self.block_norm = 'L2-Hys'
        self.target_size = (64, 64)  # Standard size for character images

    def preprocess_image(self, char_image):
        """Preprocess character image for HOG extraction"""
        # Convert to grayscale if needed
        if len(char_image.shape) == 3:
            if cv2 is not None:
                char_image = cv2.cvtColor(char_image, cv2.COLOR_BGR2GRAY)
            else:
                # Simple RGB to grayscale conversion
                char_image = np.dot(char_image[...,:3], [0.2989, 0.5870, 0.1140])

        # Resize to target size
        if transform is not None:
            resized = transform.resize(
                char_image, 
                self.target_size, 
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.uint8)
        else:
            resized = self.simple_resize(char_image, self.target_size)

        # Normalize intensity
        if resized.std() > 0:
            resized = (resized - resized.mean()) / resized.std()
            resized = np.clip(resized * 50 + 128, 0, 255).astype(np.uint8)

        return resized

    def simple_resize(self, image, target_size):
        """Simple image resizing without external libraries"""
        h, w = image.shape
        target_h, target_w = target_size

        # Calculate scaling factors
        scale_h = h / target_h
        scale_w = w / target_w

        resized = np.zeros(target_size, dtype=image.dtype)

        for y in range(target_h):
            for x in range(target_w):
                # Find corresponding pixel in original image
                orig_y = int(y * scale_h)
                orig_x = int(x * scale_w)

                # Ensure indices are within bounds
                orig_y = min(orig_y, h - 1)
                orig_x = min(orig_x, w - 1)

                resized[y, x] = image[orig_y, orig_x]

        return resized

    def compute_gradients(self, image):
        """Compute gradients for HOG calculation"""
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

        # Pad image
        padded = np.pad(image, 1, mode='edge')

        # Compute gradients
        grad_x = np.zeros_like(image, dtype=float)
        grad_y = np.zeros_like(image, dtype=float)

        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                # Apply Sobel kernels
                patch = padded[y:y+3, x:x+3]
                grad_x[y, x] = np.sum(patch * sobel_x)
                grad_y[y, x] = np.sum(patch * sobel_y)

        return grad_x, grad_y

    def compute_magnitude_and_orientation(self, grad_x, grad_y):
        """Compute gradient magnitude and orientation"""
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x) * 180 / np.pi

        # Convert to 0-180 degree range
        orientation = np.abs(orientation)

        return magnitude, orientation

    def create_histogram_cells(self, magnitude, orientation):
        """Create histogram cells for HOG"""
        h, w = magnitude.shape
        cell_h, cell_w = self.pixels_per_cell

        # Calculate number of cells
        cells_y = h // cell_h
        cells_x = w // cell_w

        # Create histogram for each cell
        histograms = np.zeros((cells_y, cells_x, self.orientations))

        # Bin width for orientation
        bin_width = 180.0 / self.orientations

        for cell_y in range(cells_y):
            for cell_x in range(cells_x):
                # Extract cell region
                y_start = cell_y * cell_h
                y_end = (cell_y + 1) * cell_h
                x_start = cell_x * cell_w
                x_end = (cell_x + 1) * cell_w

                cell_mag = magnitude[y_start:y_end, x_start:x_end]
                cell_ori = orientation[y_start:y_end, x_start:x_end]

                # Create histogram for this cell
                for y in range(cell_h):
                    for x in range(cell_w):
                        mag = cell_mag[y, x]
                        ori = cell_ori[y, x]

                        # Find bin indices
                        bin_idx = ori / bin_width

                        # Bilinear interpolation between bins
                        bin_low = int(np.floor(bin_idx)) % self.orientations
                        bin_high = (bin_low + 1) % self.orientations

                        # Interpolation weights
                        weight_high = bin_idx - np.floor(bin_idx)
                        weight_low = 1.0 - weight_high

                        # Add to histogram
                        histograms[cell_y, cell_x, bin_low] += mag * weight_low
                        histograms[cell_y, cell_x, bin_high] += mag * weight_high

        return histograms

    def normalize_blocks(self, cell_histograms):
        """Normalize histograms in blocks"""
        cells_y, cells_x = cell_histograms.shape[:2]
        block_h, block_w = self.cells_per_block

        # Calculate number of blocks
        blocks_y = cells_y - block_h + 1
        blocks_x = cells_x - block_w + 1

        if blocks_y <= 0 or blocks_x <= 0:
            # Not enough cells for blocks, return flattened histograms
            return cell_histograms.flatten()

        normalized_features = []

        for block_y in range(blocks_y):
            for block_x in range(blocks_x):
                # Extract block
                block = cell_histograms[
                    block_y:block_y+block_h, 
                    block_x:block_x+block_w
                ]

                # Flatten block
                block_vector = block.flatten()

                # Normalize block
                if self.block_norm == 'L2-Hys':
                    # L2 normalization
                    norm = np.linalg.norm(block_vector)
                    if norm > 0:
                        block_vector = block_vector / norm

                    # Clipping
                    block_vector = np.clip(block_vector, 0, 0.2)

                    # Renormalize
                    norm = np.linalg.norm(block_vector)
                    if norm > 0:
                        block_vector = block_vector / norm
                elif self.block_norm == 'L1':
                    # L1 normalization
                    norm = np.sum(np.abs(block_vector))
                    if norm > 0:
                        block_vector = block_vector / norm

                normalized_features.extend(block_vector)

        return np.array(normalized_features)

    def extract_hog_fallback(self, image):
        """Fallback HOG implementation without external libraries"""
        # Compute gradients
        grad_x, grad_y = self.compute_gradients(image.astype(float))

        # Compute magnitude and orientation
        magnitude, orientation = self.compute_magnitude_and_orientation(grad_x, grad_y)

        # Create cell histograms
        cell_histograms = self.create_histogram_cells(magnitude, orientation)

        # Normalize blocks
        hog_features = self.normalize_blocks(cell_histograms)

        return hog_features

    def extract_hog_features(self, char_image):
        """Extract HOG features from character image"""
        # Preprocess image
        processed_image = self.preprocess_image(char_image)

        if hog is not None:
            # Use scikit-image HOG implementation
            hog_features = hog(
                processed_image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                block_norm=self.block_norm,
                visualize=False,
                transform_sqrt=True
            )
        else:
            # Use fallback implementation
            hog_features = self.extract_hog_fallback(processed_image)

        return hog_features

    def extract_batch_features(self, char_images):
        """Extract HOG features from multiple character images"""
        features = []

        for char_image in char_images:
            hog_features = self.extract_hog_features(char_image)
            features.append(hog_features)

        return np.array(features)

    def get_feature_dimension(self):
        """Calculate expected feature dimension"""
        # Create dummy image to calculate feature dimension
        dummy_image = np.zeros(self.target_size, dtype=np.uint8)
        features = self.extract_hog_features(dummy_image)
        return len(features)

# Convenience function
def extract_hog_features(char_image):
    """Convenience function for HOG feature extraction"""
    extractor = HOGExtractor()
    return extractor.extract_hog_features(char_image)
