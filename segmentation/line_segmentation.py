
import numpy as np
try:
    import cv2
except ImportError:
    print("OpenCV not available - using fallback implementations")
    cv2 = None

class LineSegmenter:
    """
    Text line segmentation using horizontal projection and baseline detection
    """

    def __init__(self):
        self.min_line_height = 8
        self.min_line_width = 50
        self.line_spacing_factor = 0.3  # Minimum spacing between lines as fraction of line height

    def horizontal_projection(self, binary_region):
        """Calculate horizontal projection profile"""
        # Count black pixels in each row
        projection = np.sum(binary_region == 0, axis=1)
        return projection

    def find_text_lines_projection(self, binary_region):
        """Find text lines using horizontal projection analysis"""
        h, w = binary_region.shape
        projection = self.horizontal_projection(binary_region)

        # Smooth projection to reduce noise
        smoothed_projection = self.smooth_projection(projection)

        # Find peaks (lines with text)
        peaks = self.find_projection_peaks(smoothed_projection)

        # Convert peaks to line boundaries
        line_boundaries = self.peaks_to_boundaries(peaks, smoothed_projection, h)

        # Filter and validate lines
        valid_lines = self.validate_lines(line_boundaries, w)

        return valid_lines

    def smooth_projection(self, projection, window_size=3):
        """Apply smoothing to projection profile"""
        if len(projection) < window_size:
            return projection

        smoothed = np.convolve(projection, np.ones(window_size)/window_size, mode='same')
        return smoothed

    def find_projection_peaks(self, projection):
        """Find peaks in horizontal projection"""
        peaks = []
        threshold = np.mean(projection) * 0.1  # 10% of mean as threshold

        in_peak = False
        peak_start = 0

        for i, value in enumerate(projection):
            if not in_peak and value > threshold:
                # Start of peak
                in_peak = True
                peak_start = i
            elif in_peak and value <= threshold:
                # End of peak
                in_peak = False
                peak_center = (peak_start + i - 1) // 2
                peak_height = np.max(projection[peak_start:i])
                peaks.append({
                    'center': peak_center,
                    'start': peak_start,
                    'end': i - 1,
                    'height': peak_height
                })

        # Handle case where peak extends to end
        if in_peak:
            peak_center = (peak_start + len(projection) - 1) // 2
            peak_height = np.max(projection[peak_start:])
            peaks.append({
                'center': peak_center,
                'start': peak_start,
                'end': len(projection) - 1,
                'height': peak_height
            })

        return peaks

    def peaks_to_boundaries(self, peaks, projection, image_height):
        """Convert peaks to line boundaries"""
        if not peaks:
            return []

        boundaries = []

        for i, peak in enumerate(peaks):
            # Find valley before peak (top boundary)
            if i == 0:
                top = 0
            else:
                # Find minimum between this peak and previous peak
                prev_end = peaks[i-1]['end']
                current_start = peak['start']
                valley_region = projection[prev_end:current_start+1]
                if len(valley_region) > 0:
                    valley_idx = np.argmin(valley_region)
                    top = prev_end + valley_idx
                else:
                    top = prev_end

            # Find valley after peak (bottom boundary)
            if i == len(peaks) - 1:
                bottom = image_height - 1
            else:
                # Find minimum between this peak and next peak
                current_end = peak['end']
                next_start = peaks[i+1]['start']
                valley_region = projection[current_end:next_start+1]
                if len(valley_region) > 0:
                    valley_idx = np.argmin(valley_region)
                    bottom = current_end + valley_idx
                else:
                    bottom = current_end

            boundaries.append({
                'top': top,
                'bottom': bottom,
                'height': bottom - top + 1,
                'peak': peak
            })

        return boundaries

    def validate_lines(self, boundaries, image_width):
        """Validate and filter line boundaries"""
        valid_lines = []

        for boundary in boundaries:
            height = boundary['height']

            # Filter by minimum height
            if height >= self.min_line_height:
                # Check if line has sufficient content
                content_ratio = boundary['peak']['height'] / max(image_width * 0.1, 1)
                if content_ratio > 0.05:  # At least 5% content
                    valid_lines.append({
                        'bbox': (0, boundary['top'], image_width, height),
                        'baseline': boundary['bottom'] - int(height * 0.2),  # Estimate baseline
                        'height': height,
                        'content_density': content_ratio
                    })

        return valid_lines

    def detect_baselines(self, binary_region, line_boundaries):
        """Detect baselines for text lines using bottom contour analysis"""
        baselines = []

        for line_info in line_boundaries:
            top = line_info['bbox'][1]
            height = line_info['bbox'][3]
            bottom = top + height

            # Extract line region
            line_region = binary_region[top:bottom, :]

            if cv2 is not None:
                # Find contours in line region
                contours, _ = cv2.findContours(
                    255 - line_region,  # Invert for contour detection
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                # Find bottom points of characters
                bottom_points = []
                for contour in contours:
                    # Find bottommost point of each contour
                    if len(contour) > 0:
                        bottom_point = max(contour[:, 0, :], key=lambda p: p[1])
                        bottom_points.append(bottom_point)

                if bottom_points:
                    # Fit line to bottom points (baseline)
                    bottom_points = np.array(bottom_points)
                    if len(bottom_points) >= 2:
                        # Simple linear regression
                        x_coords = bottom_points[:, 0]
                        y_coords = bottom_points[:, 1] + top  # Convert to image coordinates

                        # Fit line y = mx + b
                        A = np.vstack([x_coords, np.ones(len(x_coords))]).T
                        m, b = np.linalg.lstsq(A, y_coords, rcond=None)[0]

                        baseline = {
                            'slope': m,
                            'intercept': b,
                            'points': bottom_points
                        }
                    else:
                        # Single point or no points, use horizontal baseline
                        baseline = {
                            'slope': 0,
                            'intercept': bottom - int(height * 0.2),
                            'points': []
                        }
                else:
                    # No contours found, estimate baseline
                    baseline = {
                        'slope': 0,
                        'intercept': bottom - int(height * 0.2),
                        'points': []
                    }
            else:
                # Fallback: simple horizontal baseline estimation
                baseline = {
                    'slope': 0,
                    'intercept': bottom - int(height * 0.2),
                    'points': []
                }

            baselines.append(baseline)

        return baselines

    def refine_line_boundaries(self, binary_region, initial_lines):
        """Refine line boundaries using content analysis"""
        refined_lines = []

        for line_info in initial_lines:
            x, y, w, h = line_info['bbox']

            # Extract line region
            line_region = binary_region[y:y+h, x:x+w]

            # Find actual content boundaries
            row_sums = np.sum(line_region == 0, axis=1)  # Black pixel counts

            # Find first and last rows with content
            content_rows = np.where(row_sums > 0)[0]

            if len(content_rows) > 0:
                actual_top = content_rows[0]
                actual_bottom = content_rows[-1]
                actual_height = actual_bottom - actual_top + 1

                if actual_height >= self.min_line_height:
                    refined_bbox = (x, y + actual_top, w, actual_height)
                    refined_lines.append({
                        'bbox': refined_bbox,
                        'baseline': y + actual_bottom - int(actual_height * 0.2),
                        'height': actual_height,
                        'content_density': line_info['content_density']
                    })

        return refined_lines

    def segment_lines(self, binary_region):
        """Main line segmentation pipeline"""
        # Find initial lines using projection
        initial_lines = self.find_text_lines_projection(binary_region)

        # Refine boundaries based on actual content
        refined_lines = self.refine_line_boundaries(binary_region, initial_lines)

        # Detect baselines
        baselines = self.detect_baselines(binary_region, refined_lines)

        # Combine line info with baselines
        final_lines = []
        for i, line_info in enumerate(refined_lines):
            line_data = line_info.copy()
            if i < len(baselines):
                line_data['baseline_info'] = baselines[i]

            final_lines.append(line_data)

        return final_lines

# Convenience function
def segment_text_lines(binary_region):
    """Convenience function for line segmentation"""
    segmenter = LineSegmenter()
    return segmenter.segment_lines(binary_region)