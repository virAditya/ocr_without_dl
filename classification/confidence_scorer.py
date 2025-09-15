import numpy as np
try:
    import pickle
except ImportError:
    print("Pickle not available")
    pickle = None

class ConfidenceScorer:
    """
    Confidence scoring system for OCR predictions with rejection thresholds
    """

    def __init__(self):
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }

        self.rejection_threshold = 0.3
        self.calibration_data = None
        self.is_calibrated = False

    def calculate_entropy_confidence(self, probabilities):
        """Calculate confidence based on entropy of probability distribution"""
        probabilities = np.array(probabilities)

        # Avoid log(0) by adding small epsilon
        epsilon = 1e-10
        probabilities = np.clip(probabilities, epsilon, 1.0)

        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities))

        # Convert to confidence (lower entropy = higher confidence)
        max_entropy = np.log2(len(probabilities))  # Maximum possible entropy
        confidence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0

        return confidence

    def calculate_margin_confidence(self, probabilities):
        """Calculate confidence based on margin between top two predictions"""
        probabilities = np.array(probabilities)

        if len(probabilities) < 2:
            return probabilities[0] if len(probabilities) == 1 else 0.0

        # Sort probabilities in descending order
        sorted_probs = np.sort(probabilities)[::-1]

        # Margin between top two predictions
        margin = sorted_probs[0] - sorted_probs[1]

        # Scale margin to confidence (0 to 1)
        # Large margin indicates high confidence
        confidence = min(1.0, margin * 2)  # Scale factor of 2

        return confidence

    def calculate_ensemble_confidence(self, individual_predictions, ensemble_prediction):
        """Calculate confidence based on agreement between ensemble members"""
        if not individual_predictions:
            return 0.0

        agreement_count = 0
        total_classifiers = 0

        for classifier_name, predictions in individual_predictions.items():
            total_classifiers += 1

            # Handle both single prediction and list of predictions
            pred = predictions[0] if isinstance(predictions, list) else predictions

            if pred == ensemble_prediction:
                agreement_count += 1

        # Agreement ratio as confidence
        confidence = agreement_count / total_classifiers if total_classifiers > 0 else 0.0

        return confidence

    def calculate_feature_quality_confidence(self, feature_vector):
        """Calculate confidence based on feature vector quality"""
        feature_vector = np.array(feature_vector)

        if len(feature_vector) == 0:
            return 0.0

        # Feature quality indicators
        quality_scores = []

        # 1. Feature completeness (non-zero features ratio)
        non_zero_ratio = np.count_nonzero(feature_vector) / len(feature_vector)
        quality_scores.append(non_zero_ratio)

        # 2. Feature variance (higher variance often indicates better features)
        if len(feature_vector) > 1:
            feature_variance = np.var(feature_vector)
            # Normalize variance score (sigmoid-like function)
            variance_score = 2 / (1 + np.exp(-feature_variance)) - 1
            quality_scores.append(max(0, variance_score))

        # 3. Feature magnitude (reasonable range check)
        feature_magnitudes = np.abs(feature_vector)
        reasonable_magnitude = np.mean(
            (feature_magnitudes >= 0.01) & (feature_magnitudes <= 100)
        )
        quality_scores.append(reasonable_magnitude)

        # Combine quality scores
        overall_quality = np.mean(quality_scores)

        return overall_quality

    def calculate_geometric_confidence(self, character_bbox, line_height=None):
        """Calculate confidence based on character geometry"""
        if not character_bbox or len(character_bbox) < 4:
            return 0.5  # Neutral confidence for missing geometry

        x, y, width, height = character_bbox

        confidence_factors = []

        # 1. Aspect ratio reasonableness
        if height > 0:
            aspect_ratio = width / height
            # Reasonable aspect ratios for characters: 0.2 to 3.0
            if 0.2 <= aspect_ratio <= 3.0:
                confidence_factors.append(1.0)
            else:
                # Penalize extreme aspect ratios
                penalty = min(abs(aspect_ratio - 0.6), 2.0) / 2.0  # 0.6 is typical
                confidence_factors.append(max(0.2, 1.0 - penalty))

        # 2. Size reasonableness
        char_area = width * height
        if char_area > 0:
            # Reasonable character sizes (very context dependent)
            if 50 <= char_area <= 5000:  # Pixels
                confidence_factors.append(1.0)
            else:
                # Penalize very small or very large characters
                size_score = max(0.3, 1.0 - abs(np.log10(char_area / 500)) / 2)
                confidence_factors.append(size_score)

        # 3. Consistency with line height (if available)
        if line_height and line_height > 0:
            height_ratio = height / line_height
            # Characters should be reasonably sized relative to line height
            if 0.5 <= height_ratio <= 1.2:
                confidence_factors.append(1.0)
            else:
                height_consistency = max(0.3, 1.0 - abs(height_ratio - 0.8) / 0.5)
                confidence_factors.append(height_consistency)

        # Combine geometric confidence factors
        if confidence_factors:
            geometric_confidence = np.mean(confidence_factors)
        else:
            geometric_confidence = 0.5

        return geometric_confidence

    def combine_confidence_scores(self, scores, weights=None):
        """Combine multiple confidence scores with optional weights"""
        if not scores:
            return 0.0

        scores = np.array(scores)

        if weights is None:
            # Equal weights
            combined_confidence = np.mean(scores)
        else:
            weights = np.array(weights)
            if len(weights) != len(scores):
                # Fallback to equal weights if dimensions don't match
                combined_confidence = np.mean(scores)
            else:
                # Weighted average
                weights = weights / np.sum(weights)  # Normalize weights
                combined_confidence = np.sum(scores * weights)

        return np.clip(combined_confidence, 0.0, 1.0)

    def score_prediction(self, prediction_data):
        """
        Score a prediction with comprehensive confidence analysis

        prediction_data should contain:
        - probabilities: class probabilities from classifier
        - individual_predictions: predictions from ensemble members
        - ensemble_prediction: final ensemble prediction
        - feature_vector: input features used for prediction
        - character_bbox: bounding box of character (optional)
        - line_height: height of text line (optional)
        """

        confidence_scores = []
        score_weights = []

        # 1. Probability-based confidence
        if 'probabilities' in prediction_data:
            prob_confidence = max(prediction_data['probabilities'])
            entropy_confidence = self.calculate_entropy_confidence(
                prediction_data['probabilities']
            )
            margin_confidence = self.calculate_margin_confidence(
                prediction_data['probabilities']
            )

            # Combine probability-based scores
            prob_score = np.mean([prob_confidence, entropy_confidence, margin_confidence])
            confidence_scores.append(prob_score)
            score_weights.append(0.4)  # High weight for probability-based confidence

        # 2. Ensemble agreement confidence
        if ('individual_predictions' in prediction_data and 
            'ensemble_prediction' in prediction_data):
            ensemble_confidence = self.calculate_ensemble_confidence(
                prediction_data['individual_predictions'],
                prediction_data['ensemble_prediction']
            )
            confidence_scores.append(ensemble_confidence)
            score_weights.append(0.3)  # Medium weight for ensemble agreement

        # 3. Feature quality confidence
        if 'feature_vector' in prediction_data:
            feature_confidence = self.calculate_feature_quality_confidence(
                prediction_data['feature_vector']
            )
            confidence_scores.append(feature_confidence)
            score_weights.append(0.2)  # Lower weight for feature quality

        # 4. Geometric confidence
        geometric_confidence = self.calculate_geometric_confidence(
            prediction_data.get('character_bbox'),
            prediction_data.get('line_height')
        )
        confidence_scores.append(geometric_confidence)
        score_weights.append(0.1)  # Lowest weight for geometric factors

        # Combine all confidence scores
        overall_confidence = self.combine_confidence_scores(
            confidence_scores, score_weights
        )

        return {
            'overall_confidence': overall_confidence,
            'confidence_level': self.get_confidence_level(overall_confidence),
            'should_reject': overall_confidence < self.rejection_threshold,
            'component_scores': {
                'probability_based': confidence_scores[0] if len(confidence_scores) > 0 else None,
                'ensemble_agreement': confidence_scores[1] if len(confidence_scores) > 1 else None,
                'feature_quality': confidence_scores[2] if len(confidence_scores) > 2 else None,
                'geometric': confidence_scores[3] if len(confidence_scores) > 3 else None
            }
        }

    def get_confidence_level(self, confidence_score):
        """Convert numerical confidence to categorical level"""
        if confidence_score >= self.confidence_thresholds['high']:
            return 'high'
        elif confidence_score >= self.confidence_thresholds['medium']:
            return 'medium'
        elif confidence_score >= self.confidence_thresholds['low']:
            return 'low'
        else:
            return 'very_low'

    def should_reject_prediction(self, confidence_score):
        """Determine if prediction should be rejected based on confidence"""
        return confidence_score < self.rejection_threshold

    def calibrate_confidence(self, validation_predictions, validation_ground_truth):
        """Calibrate confidence scores using validation data"""
        # This would implement confidence calibration techniques like Platt scaling
        # For now, we'll store the calibration data for future use

        self.calibration_data = {
            'predictions': validation_predictions,
            'ground_truth': validation_ground_truth
        }

        # Simple calibration: adjust thresholds based on validation accuracy
        correct_predictions = np.array(validation_predictions) == np.array(validation_ground_truth)
        accuracy = np.mean(correct_predictions)

        # Adjust rejection threshold based on accuracy
        if accuracy < 0.7:
            self.rejection_threshold = min(0.5, self.rejection_threshold + 0.1)
        elif accuracy > 0.9:
            self.rejection_threshold = max(0.2, self.rejection_threshold - 0.05)

        self.is_calibrated = True

        return {
            'validation_accuracy': accuracy,
            'adjusted_rejection_threshold': self.rejection_threshold
        }

    def set_confidence_thresholds(self, high=None, medium=None, low=None, rejection=None):
        """Set custom confidence thresholds"""
        if high is not None:
            self.confidence_thresholds['high'] = high
        if medium is not None:
            self.confidence_thresholds['medium'] = medium
        if low is not None:
            self.confidence_thresholds['low'] = low
        if rejection is not None:
            self.rejection_threshold = rejection

    def save_scorer(self, filepath):
        """Save confidence scorer configuration"""
        if pickle is None:
            print("Pickle not available - cannot save scorer")
            return

        scorer_data = {
            'confidence_thresholds': self.confidence_thresholds,
            'rejection_threshold': self.rejection_threshold,
            'calibration_data': self.calibration_data,
            'is_calibrated': self.is_calibrated
        }

        try:
            with open(filepath, 'wb') as f:
                pickle.dump(scorer_data, f)
            print(f"Confidence scorer saved to {filepath}")
        except Exception as e:
            print(f"Error saving scorer: {e}")

    def load_scorer(self, filepath):
        """Load confidence scorer configuration"""
        if pickle is None:
            print("Pickle not available - cannot load scorer")
            return

        try:
            with open(filepath, 'rb') as f:
                scorer_data = pickle.load(f)

            self.confidence_thresholds = scorer_data.get('confidence_thresholds', self.confidence_thresholds)
            self.rejection_threshold = scorer_data.get('rejection_threshold', self.rejection_threshold)
            self.calibration_data = scorer_data.get('calibration_data')
            self.is_calibrated = scorer_data.get('is_calibrated', False)

            print(f"Confidence scorer loaded from {filepath}")

        except Exception as e:
            print(f"Error loading scorer: {e}")

# Convenience function
def create_confidence_scorer(high_threshold=0.8, medium_threshold=0.6, 
                           low_threshold=0.4, rejection_threshold=0.3):
    """Convenience function for creating confidence scorer"""
    scorer = ConfidenceScorer()
    scorer.set_confidence_thresholds(high_threshold, medium_threshold, 
                                   low_threshold, rejection_threshold)
    return scorer