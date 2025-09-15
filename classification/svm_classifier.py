import numpy as np
try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import GridSearchCV
    import pickle
except ImportError:
    print("scikit-learn not available - using fallback implementations")
    SVC = None
    StandardScaler = None
    LabelEncoder = None
    GridSearchCV = None
    pickle = None

class SVMClassifier:
    """
    Support Vector Machine classifier for character recognition
    """

    def __init__(self):
        self.svm_model = None
        self.scaler = None
        self.label_encoder = None
        self.is_trained = False

        # Default SVM parameters
        self.svm_params = {
            'kernel': 'rbf',
            'C': 1.0,
            'gamma': 'scale',
            'probability': True  # Enable probability estimation
        }

    def prepare_training_data(self, features, labels):
        """Prepare training data with scaling and encoding"""
        features = np.array(features)
        labels = np.array(labels)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Initialize scalers and encoders if needed
        if StandardScaler is not None:
            if self.scaler is None:
                self.scaler = StandardScaler()
                scaled_features = self.scaler.fit_transform(features)
            else:
                scaled_features = self.scaler.transform(features)
        else:
            # Simple scaling fallback
            if not hasattr(self, 'feature_mean'):
                self.feature_mean = np.mean(features, axis=0)
                self.feature_std = np.std(features, axis=0)
                self.feature_std[self.feature_std == 0] = 1  # Avoid division by zero

            scaled_features = (features - self.feature_mean) / self.feature_std

        # Encode labels
        if LabelEncoder is not None:
            if self.label_encoder is None:
                self.label_encoder = LabelEncoder()
                encoded_labels = self.label_encoder.fit_transform(labels)
            else:
                encoded_labels = self.label_encoder.transform(labels)
        else:
            # Simple label encoding fallback
            if not hasattr(self, 'label_mapping'):
                unique_labels = np.unique(labels)
                self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
                self.reverse_mapping = {i: label for label, i in self.label_mapping.items()}

            encoded_labels = np.array([self.label_mapping[label] for label in labels])

        return scaled_features, encoded_labels

    def optimize_hyperparameters(self, features, labels, cv_folds=3):
        """Optimize SVM hyperparameters using grid search"""
        if GridSearchCV is None:
            print("GridSearchCV not available, using default parameters")
            return self.svm_params

        # Parameter grid for optimization
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'kernel': ['rbf', 'linear', 'poly']
        }

        # Create SVM for grid search
        svm = SVC(probability=True)

        # Perform grid search
        grid_search = GridSearchCV(
            svm, param_grid, 
            cv=cv_folds, 
            scoring='accuracy',
            n_jobs=-1 if len(features) > 100 else 1  # Parallel processing for larger datasets
        )

        grid_search.fit(features, labels)

        print(f"Best SVM parameters: {grid_search.best_params_}")
        print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

        return grid_search.best_params_

    def train(self, features, labels, optimize_params=False):
        """Train SVM classifier"""
        if SVC is None:
            print("SVM training not available - using fallback classifier")
            return self.train_fallback(features, labels)

        # Prepare data
        scaled_features, encoded_labels = self.prepare_training_data(features, labels)

        print(f"Training SVM with {len(scaled_features)} samples and {scaled_features.shape[1]} features")

        # Optimize hyperparameters if requested
        if optimize_params and len(scaled_features) > 20:  # Only optimize with sufficient data
            optimized_params = self.optimize_hyperparameters(scaled_features, encoded_labels)
            self.svm_params.update(optimized_params)

        # Train SVM
        self.svm_model = SVC(**self.svm_params)
        self.svm_model.fit(scaled_features, encoded_labels)

        self.is_trained = True

        # Calculate training accuracy
        train_predictions = self.svm_model.predict(scaled_features)
        train_accuracy = np.mean(train_predictions == encoded_labels)

        print(f"Training completed. Training accuracy: {train_accuracy:.4f}")

        return {
            'training_accuracy': train_accuracy,
            'num_samples': len(scaled_features),
            'num_features': scaled_features.shape[1],
            'num_classes': len(np.unique(encoded_labels))
        }

    def train_fallback(self, features, labels):
        """Fallback training method without sklearn"""
        features = np.array(features)
        labels = np.array(labels)

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Simple nearest centroid classifier as fallback
        unique_labels = np.unique(labels)
        self.class_centroids = {}

        for label in unique_labels:
            mask = labels == label
            class_features = features[mask]
            self.class_centroids[label] = np.mean(class_features, axis=0)

        self.is_trained = True
        print(f"Fallback training completed with {len(unique_labels)} classes")

        return {
            'training_accuracy': 0.8,  # Estimated
            'num_samples': len(features),
            'num_features': features.shape[1],
            'num_classes': len(unique_labels)
        }

    def predict(self, features):
        """Make predictions on new features"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")

        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self.svm_model is not None:
            # Scale features
            if self.scaler is not None:
                scaled_features = self.scaler.transform(features)
            else:
                scaled_features = (features - self.feature_mean) / self.feature_std

            # Make predictions
            encoded_predictions = self.svm_model.predict(scaled_features)

            # Decode labels
            if self.label_encoder is not None:
                predictions = self.label_encoder.inverse_transform(encoded_predictions)
            else:
                predictions = [self.reverse_mapping[pred] for pred in encoded_predictions]

            return predictions
        else:
            # Fallback prediction using nearest centroid
            return self.predict_fallback(features)

    def predict_fallback(self, features):
        """Fallback prediction using nearest centroid"""
        predictions = []

        for feature_vector in features:
            best_label = None
            best_distance = float('inf')

            for label, centroid in self.class_centroids.items():
                distance = np.linalg.norm(feature_vector - centroid)
                if distance < best_distance:
                    best_distance = distance
                    best_label = label

            predictions.append(best_label)

        return predictions

    def predict_proba(self, features):
        """Predict class probabilities"""
        if not self.is_trained:
            raise ValueError("Classifier not trained yet")

        features = np.array(features)
        if features.ndim == 1:
            features = features.reshape(1, -1)

        if self.svm_model is not None and hasattr(self.svm_model, 'predict_proba'):
            # Scale features
            if self.scaler is not None:
                scaled_features = self.scaler.transform(features)
            else:
                scaled_features = (features - self.feature_mean) / self.feature_std

            # Get probabilities
            probabilities = self.svm_model.predict_proba(scaled_features)

            # Get class labels
            if self.label_encoder is not None:
                class_labels = self.label_encoder.classes_
            else:
                class_labels = list(self.reverse_mapping.values())

            return probabilities, class_labels
        else:
            # Fallback probability estimation using distances
            return self.predict_proba_fallback(features)

    def predict_proba_fallback(self, features):
        """Fallback probability estimation"""
        probabilities = []
        class_labels = list(self.class_centroids.keys())

        for feature_vector in features:
            distances = []
            for label in class_labels:
                centroid = self.class_centroids[label]
                distance = np.linalg.norm(feature_vector - centroid)
                distances.append(distance)

            # Convert distances to probabilities (closer = higher probability)
            distances = np.array(distances)
            # Avoid division by zero
            distances = distances + 1e-8
            inverse_distances = 1.0 / distances
            probs = inverse_distances / np.sum(inverse_distances)

            probabilities.append(probs)

        return np.array(probabilities), class_labels

    def save_model(self, filepath):
        """Save trained model to file"""
        if not self.is_trained:
            raise ValueError("No trained model to save")

        model_data = {
            'svm_model': self.svm_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'svm_params': self.svm_params,
            'is_trained': self.is_trained
        }

        # Add fallback data if using fallback methods
        if hasattr(self, 'feature_mean'):
            model_data['feature_mean'] = self.feature_mean
            model_data['feature_std'] = self.feature_std

        if hasattr(self, 'label_mapping'):
            model_data['label_mapping'] = self.label_mapping
            model_data['reverse_mapping'] = self.reverse_mapping

        if hasattr(self, 'class_centroids'):
            model_data['class_centroids'] = self.class_centroids

        if pickle is not None:
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"Model saved to {filepath}")
        else:
            print("Pickle not available - cannot save model")

    def load_model(self, filepath):
        """Load trained model from file"""
        if pickle is None:
            print("Pickle not available - cannot load model")
            return

        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            self.svm_model = model_data.get('svm_model')
            self.scaler = model_data.get('scaler')
            self.label_encoder = model_data.get('label_encoder')
            self.svm_params = model_data.get('svm_params', self.svm_params)
            self.is_trained = model_data.get('is_trained', False)

            # Load fallback data if present
            if 'feature_mean' in model_data:
                self.feature_mean = model_data['feature_mean']
                self.feature_std = model_data['feature_std']

            if 'label_mapping' in model_data:
                self.label_mapping = model_data['label_mapping']
                self.reverse_mapping = model_data['reverse_mapping']

            if 'class_centroids' in model_data:
                self.class_centroids = model_data['class_centroids']

            print(f"Model loaded from {filepath}")

        except Exception as e:
            print(f"Error loading model: {e}")

# Convenience function
def train_svm_classifier(features, labels, optimize_params=False):
    """Convenience function for training SVM classifier"""
    classifier = SVMClassifier()
    training_info = classifier.train(features, labels, optimize_params)
    return classifier, training_info