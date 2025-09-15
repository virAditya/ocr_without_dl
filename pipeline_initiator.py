#!/usr/bin/env python3
"""
OCR Pipeline Initiator

Main entry point for the Classical Computer Vision OCR Pipeline.
This script orchestrates the entire OCR process from image input to text output.
"""

import os
import sys
import numpy as np
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all pipeline components
try:
    # Preprocessing modules
    from preprocessing.binarization import ImageBinarizer
    from preprocessing.deskewing import ImageDeskewer
    from preprocessing.noise_removal import NoiseRemover

    # Segmentation modules
    from segmentation.layout_analysis import LayoutAnalyzer
    from segmentation.line_segmentation import LineSegmenter
    from segmentation.character_segmentation import CharacterSegmenter

    # Feature extraction modules
    from features.hog_extractor import HOGExtractor
    from features.structural_features import StructuralFeatureExtractor
    from features.geometric_features import GeometricFeatureExtractor

    # Classification modules
    from classification.svm_classifier import SVMClassifier
    from classification.ensemble_classifier import EnsembleClassifier
    from classification.confidence_scorer import ConfidenceScorer

    print("All modules imported successfully!")

except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available, using fallback implementations where possible")

class OCRPipeline:
    """
    Complete OCR Pipeline using Classical Computer Vision techniques
    """

    def __init__(self, config=None):
        """Initialize OCR pipeline with configuration"""
        self.config = config or self.get_default_config()

        # Initialize all components
        self.binarizer = ImageBinarizer()
        self.deskewer = ImageDeskewer()
        self.noise_remover = NoiseRemover()
        self.layout_analyzer = LayoutAnalyzer()
        self.line_segmenter = LineSegmenter()
        self.character_segmenter = CharacterSegmenter()

        # Feature extractors
        self.hog_extractor = HOGExtractor()
        self.structural_extractor = StructuralFeatureExtractor()
        self.geometric_extractor = GeometricFeatureExtractor()

        # Classifiers
        self.svm_classifier = SVMClassifier()
        self.ensemble_classifier = None
        self.confidence_scorer = ConfidenceScorer()

        # Pipeline state
        self.is_trained = False
        self.processing_stats = {}

        print("OCR Pipeline initialized successfully!")

    def get_default_config(self):
        """Get default pipeline configuration"""
        return {
            'preprocessing': {
                'apply_binarization': True,
                'apply_deskewing': True,
                'apply_noise_removal': True
            },
            'segmentation': {
                'use_layout_analysis': True,
                'segment_lines': True,
                'segment_characters': True
            },
            'feature_extraction': {
                'use_hog': True,
                'use_structural': True,
                'use_geometric': True,
                'feature_weights': [0.5, 0.3, 0.2]  # HOG, Structural, Geometric
            },
            'classification': {
                'use_ensemble': True,
                'confidence_scoring': True,
                'rejection_threshold': 0.3
            },
            'output': {
                'save_intermediate_results': False,
                'output_confidence_scores': True,
                'output_bounding_boxes': True
            }
        }

    def preprocess_image(self, image_path):
        """Apply preprocessing steps to input image"""
        print(f"\nPreprocessing image: {image_path}")

        start_time = time.time()

        # Step 1: Binarization
        if self.config['preprocessing']['apply_binarization']:
            print("  - Applying binarization...")
            binary_image = self.binarizer.binarize(image_path)
        else:
            # Load image directly
            try:
                import cv2
                binary_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            except:
                from PIL import Image
                pil_image = Image.open(image_path).convert('L')
                binary_image = np.array(pil_image)

        # Step 2: Deskewing
        if self.config['preprocessing']['apply_deskewing']:
            print("  - Applying deskewing...")
            deskew_result = self.deskewer.deskew(binary_image)
            binary_image = deskew_result['corrected_image']
            print(f"    Detected angle: {deskew_result['detected_angle']:.2f} degrees")

        # Step 3: Noise removal
        if self.config['preprocessing']['apply_noise_removal']:
            print("  - Applying noise removal...")
            binary_image = self.noise_remover.clean_noise(binary_image)

        preprocessing_time = time.time() - start_time
        self.processing_stats['preprocessing_time'] = preprocessing_time
        print(f"  Preprocessing completed in {preprocessing_time:.2f} seconds")

        return binary_image

    def segment_image(self, binary_image):
        """Segment image into layout, lines, and characters"""
        print("\nSegmenting image...")

        start_time = time.time()
        segmentation_results = {
            'layout_regions': [],
            'text_lines': [],
            'characters': []
        }

        # Step 1: Layout analysis
        if self.config['segmentation']['use_layout_analysis']:
            print("  - Analyzing page layout...")
            layout_result = self.layout_analyzer.analyze_layout(binary_image)
            segmentation_results['layout_regions'] = layout_result['layout_regions']
            print(f"    Found {len(layout_result['layout_regions'])} layout regions")
        else:
            # Use entire image as single region
            h, w = binary_image.shape
            segmentation_results['layout_regions'] = [(0, 0, w, h)]

        # Step 2: Line segmentation
        if self.config['segmentation']['segment_lines']:
            print("  - Segmenting text lines...")
            for region_bbox in segmentation_results['layout_regions']:
                x, y, w, h = region_bbox
                region_image = binary_image[y:y+h, x:x+w]

                lines = self.line_segmenter.segment_lines(region_image)

                # Adjust line coordinates to global image coordinates
                for line in lines:
                    line_x, line_y, line_w, line_h = line['bbox']
                    global_bbox = (x + line_x, y + line_y, line_w, line_h)
                    line['global_bbox'] = global_bbox

                segmentation_results['text_lines'].extend(lines)

            print(f"    Found {len(segmentation_results['text_lines'])} text lines")

        # Step 3: Character segmentation
        if self.config['segmentation']['segment_characters']:
            print("  - Segmenting characters...")
            total_characters = 0

            for line in segmentation_results['text_lines']:
                if 'global_bbox' in line:
                    x, y, w, h = line['global_bbox']
                else:
                    x, y, w, h = line['bbox']

                line_image = binary_image[y:y+h, x:x+w]
                characters = self.character_segmenter.segment_characters(line_image)

                # Adjust character coordinates and add line reference
                for char in characters:
                    char_x, char_y, char_w, char_h = char['bbox']
                    global_bbox = (x + char_x, y + char_y, char_w, char_h)
                    char['global_bbox'] = global_bbox
                    char['line_info'] = line

                segmentation_results['characters'].extend(characters)
                total_characters += len(characters)

            print(f"    Found {total_characters} characters")

        segmentation_time = time.time() - start_time
        self.processing_stats['segmentation_time'] = segmentation_time
        print(f"  Segmentation completed in {segmentation_time:.2f} seconds")

        return segmentation_results

    def extract_features(self, characters):
        """Extract features from character images"""
        print("\nExtracting features...")

        start_time = time.time()
        feature_vectors = []

        for i, char_data in enumerate(characters):
            char_image = char_data['image']
            char_features = []

            # HOG features
            if self.config['feature_extraction']['use_hog']:
                hog_features = self.hog_extractor.extract_hog_features(char_image)
                char_features.append(hog_features)

            # Structural features
            if self.config['feature_extraction']['use_structural']:
                structural_features = self.structural_extractor.extract_structural_features(char_image)
                char_features.append(structural_features)

            # Geometric features
            if self.config['feature_extraction']['use_geometric']:
                geometric_features = self.geometric_extractor.extract_geometric_features(char_image)
                char_features.append(geometric_features)

            # Combine features
            if char_features:
                combined_features = np.concatenate(char_features)
                feature_vectors.append(combined_features)

            if (i + 1) % 50 == 0:
                print(f"    Processed {i + 1}/{len(characters)} characters")

        feature_extraction_time = time.time() - start_time
        self.processing_stats['feature_extraction_time'] = feature_extraction_time
        print(f"  Feature extraction completed in {feature_extraction_time:.2f} seconds")
        print(f"  Feature vector dimension: {len(feature_vectors[0]) if feature_vectors else 0}")

        return np.array(feature_vectors) if feature_vectors else np.array([])

    def recognize_characters(self, feature_vectors, characters):
        """Recognize characters using trained classifiers"""
        print("\nRecognizing characters...")

        if not self.is_trained:
            print("  Warning: No trained classifier available!")
            print("  Using dummy predictions for demonstration")
            return self.generate_dummy_predictions(len(feature_vectors))

        start_time = time.time()
        recognition_results = []

        if self.config['classification']['use_ensemble'] and self.ensemble_classifier:
            # Use ensemble classifier
            predictions = self.ensemble_classifier.predict(feature_vectors)

            if self.config['classification']['confidence_scoring']:
                # Get detailed predictions with confidence
                for i, features in enumerate(feature_vectors):
                    individual_preds = self.ensemble_classifier.predict_individual([features])
                    ensemble_pred = predictions[i] if hasattr(predictions, '__iter__') else predictions

                    # Get probabilities if available
                    try:
                        probabilities, _ = self.svm_classifier.predict_proba([features])
                        prob_vector = probabilities[0] if len(probabilities) > 0 else [0.5, 0.5]
                    except:
                        prob_vector = [0.5, 0.5]  # Default probabilities

                    # Score confidence
                    confidence_data = {
                        'probabilities': prob_vector,
                        'individual_predictions': individual_preds,
                        'ensemble_prediction': ensemble_pred,
                        'feature_vector': features,
                        'character_bbox': characters[i].get('global_bbox'),
                        'line_height': characters[i].get('line_info', {}).get('height')
                    }

                    confidence_result = self.confidence_scorer.score_prediction(confidence_data)

                    recognition_results.append({
                        'character': ensemble_pred,
                        'confidence': confidence_result['overall_confidence'],
                        'confidence_level': confidence_result['confidence_level'],
                        'should_reject': confidence_result['should_reject'],
                        'bbox': characters[i].get('global_bbox'),
                        'individual_predictions': individual_preds
                    })
        else:
            # Use SVM classifier only
            predictions = self.svm_classifier.predict(feature_vectors)

            for i, prediction in enumerate(predictions):
                recognition_results.append({
                    'character': prediction,
                    'confidence': 0.7,  # Default confidence
                    'confidence_level': 'medium',
                    'should_reject': False,
                    'bbox': characters[i].get('global_bbox')
                })

        recognition_time = time.time() - start_time
        self.processing_stats['recognition_time'] = recognition_time
        print(f"  Character recognition completed in {recognition_time:.2f} seconds")

        return recognition_results

    def generate_dummy_predictions(self, num_characters):
        """Generate dummy predictions for demonstration when no trained model exists"""
        dummy_chars = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789')

        results = []
        for i in range(num_characters):
            char = np.random.choice(dummy_chars)
            results.append({
                'character': char,
                'confidence': np.random.uniform(0.4, 0.9),
                'confidence_level': 'medium',
                'should_reject': np.random.random() < 0.1,  # 10% rejection rate
                'bbox': (0, 0, 20, 30)  # Dummy bbox
            })

        return results

    def reconstruct_text(self, recognition_results, segmentation_results):
        """Reconstruct text from character recognition results"""
        print("\nReconstructing text...")

        if not recognition_results:
            return ""

        # Group characters by lines
        lines_dict = {}

        for i, result in enumerate(recognition_results):
            # Find which line this character belongs to
            char_bbox = result['bbox']
            if not char_bbox:
                continue

            char_x, char_y, char_w, char_h = char_bbox
            char_center_y = char_y + char_h // 2

            # Find the closest line
            best_line_idx = 0
            min_distance = float('inf')

            for j, line in enumerate(segmentation_results['text_lines']):
                line_bbox = line.get('global_bbox', line['bbox'])
                line_y = line_bbox[1]
                line_h = line_bbox[3]
                line_center_y = line_y + line_h // 2

                distance = abs(char_center_y - line_center_y)
                if distance < min_distance:
                    min_distance = distance
                    best_line_idx = j

            if best_line_idx not in lines_dict:
                lines_dict[best_line_idx] = []

            lines_dict[best_line_idx].append({
                'char': result['character'],
                'x': char_x,
                'confidence': result['confidence'],
                'should_reject': result['should_reject']
            })

        # Sort characters within each line by x-coordinate
        text_lines = []
        for line_idx in sorted(lines_dict.keys()):
            chars_in_line = sorted(lines_dict[line_idx], key=lambda c: c['x'])

            # Build line text
            line_text = ""
            prev_x = 0

            for char_data in chars_in_line:
                # Add space if there's a significant gap
                if prev_x > 0 and char_data['x'] - prev_x > 30:  # Adjust threshold as needed
                    line_text += " "

                # Add character (or placeholder if rejected)
                if char_data['should_reject']:
                    line_text += "?"  # Placeholder for rejected characters
                else:
                    line_text += char_data['char']

                prev_x = char_data['x']

            text_lines.append(line_text.strip())

        # Join lines with newlines
        final_text = "\n".join(text_lines)

        print(f"  Text reconstruction completed")
        print(f"  Reconstructed {len(text_lines)} lines of text")

        return final_text

    def process_image(self, image_path, output_path=None):
        """Process a single image through the complete OCR pipeline"""
        print(f"\n{'='*60}")
        print(f"CLASSICAL OCR PIPELINE - PROCESSING: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        total_start_time = time.time()

        try:
            # Step 1: Preprocessing
            binary_image = self.preprocess_image(image_path)

            # Step 2: Segmentation
            segmentation_results = self.segment_image(binary_image)

            # Step 3: Feature extraction
            if segmentation_results['characters']:
                feature_vectors = self.extract_features(segmentation_results['characters'])

                # Step 4: Character recognition
                recognition_results = self.recognize_characters(
                    feature_vectors, segmentation_results['characters']
                )

                # Step 5: Text reconstruction
                final_text = self.reconstruct_text(recognition_results, segmentation_results)
            else:
                print("  No characters found in image!")
                recognition_results = []
                final_text = ""

            total_time = time.time() - total_start_time
            self.processing_stats['total_time'] = total_time

            # Prepare results
            results = {
                'text': final_text,
                'recognition_results': recognition_results,
                'segmentation_results': segmentation_results,
                'processing_stats': self.processing_stats,
                'image_path': image_path
            }

            # Save output if requested
            if output_path:
                self.save_results(results, output_path)

            # Print summary
            self.print_summary(results)

            return results

        except Exception as e:
            print(f"  Error processing image: {e}")
            import traceback
            traceback.print_exc()
            return None

    def save_results(self, results, output_path):
        """Save OCR results to file"""
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Save text output
            text_path = output_path
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(results['text'])

            print(f"  Results saved to: {text_path}")

        except Exception as e:
            print(f"  Error saving results: {e}")

    def print_summary(self, results):
        """Print processing summary"""
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")

        stats = results['processing_stats']

        print(f"Total processing time: {stats.get('total_time', 0):.2f} seconds")
        print(f"  - Preprocessing: {stats.get('preprocessing_time', 0):.2f}s")
        print(f"  - Segmentation: {stats.get('segmentation_time', 0):.2f}s") 
        print(f"  - Feature extraction: {stats.get('feature_extraction_time', 0):.2f}s")
        print(f"  - Recognition: {stats.get('recognition_time', 0):.2f}s")

        print(f"\nCharacters found: {len(results['recognition_results'])}")
        print(f"Text lines: {len(results['segmentation_results']['text_lines'])}")

        if results['text']:
            print(f"\nEXTRACTED TEXT:")
            print("-" * 40)
            print(results['text'])
            print("-" * 40)
        else:
            print("\nNo text extracted from image")

        print(f"\n{'='*60}")

def main():
    """Main function for command-line usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Classical Computer Vision OCR Pipeline')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', help='Output text file path')
    parser.add_argument('--config', help='Configuration file path (JSON)')

    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        return 1

    # Load configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        try:
            import json
            with open(args.config, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {args.config}")
        except Exception as e:
            print(f"Error loading configuration: {e}")

    # Initialize and run pipeline
    try:
        pipeline = OCRPipeline(config)
        results = pipeline.process_image(args.input, args.output)

        if results:
            return 0  # Success
        else:
            return 1  # Failure

    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())