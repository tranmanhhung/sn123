# MIT License - MANTIS ML Prediction Model

import logging
import pickle
import os
from typing import List, Optional, Tuple
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from miner_config import config

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Ensemble ML model for Bitcoin price prediction"""
    
    def __init__(self):
        self.models = {
            'rf': RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            ),
            'ridge': Ridge(alpha=1.0)
        }
        
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_history = []
        self.target_history = []
        self.max_history = 1000  # Keep last 1000 samples for training
        
        # Model weights for ensemble
        self.model_weights = {'rf': 0.4, 'gb': 0.4, 'ridge': 0.2}
        
        # Load existing model if available
        self.load_model()
    
    def add_training_sample(self, features: np.ndarray, target: float):
        """Add a new training sample to the history"""
        self.feature_history.append(features.copy())
        self.target_history.append(target)
        
        # Keep only recent samples
        if len(self.feature_history) > self.max_history:
            self.feature_history = self.feature_history[-self.max_history:]
            self.target_history = self.target_history[-self.max_history:]
        
        logger.info(f"Training history size: {len(self.feature_history)}")
    
    def train(self, min_samples: int = 50) -> bool:
        """Train the ensemble model"""
        if len(self.feature_history) < min_samples:
            logger.warning(f"Not enough training samples: {len(self.feature_history)} < {min_samples}")
            return False
        
        try:
            # Prepare training data
            X = np.array(self.feature_history)
            y = np.array(self.target_history)
            
            # Remove any NaN or infinite values
            valid_idx = np.isfinite(X).all(axis=1) & np.isfinite(y)
            X = X[valid_idx]
            y = y[valid_idx]
            
            if len(X) < min_samples:
                logger.warning(f"Not enough valid samples after cleaning: {len(X)}")
                return False
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Train each model
            for name, model in self.models.items():
                logger.info(f"Training {name} model...")
                model.fit(X_scaled, y)
            
            self.is_trained = True
            logger.info("Ensemble model training completed")
            
            # Calculate training metrics
            self._calculate_training_metrics(X_scaled, y)
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def predict(self, features: np.ndarray) -> Tuple[float, float]:
        """Make prediction using ensemble model
        
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_trained:
            logger.warning("Model not trained, returning neutral prediction")
            return 0.0, 0.0
        
        try:
            # Ensure features are the right shape
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Get predictions from each model
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(features_scaled)[0]
                predictions.append(pred * self.model_weights[name])
            
            # Ensemble prediction
            ensemble_pred = sum(predictions)
            
            # Calculate confidence based on agreement between models
            individual_preds = [model.predict(features_scaled)[0] for model in self.models.values()]
            confidence = 1.0 / (1.0 + np.std(individual_preds))
            
            # Clip prediction to reasonable range
            ensemble_pred = np.clip(ensemble_pred, -0.1, 0.1)  # Max 10% price change
            
            logger.debug(f"Prediction: {ensemble_pred:.4f}, Confidence: {confidence:.4f}")
            
            return float(ensemble_pred), float(confidence)
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0.0, 0.0
    
    def _calculate_training_metrics(self, X: np.ndarray, y: np.ndarray):
        """Calculate and log training metrics"""
        try:
            predictions = []
            for name, model in self.models.items():
                pred = model.predict(X)
                predictions.append(pred * self.model_weights[name])
            
            ensemble_pred = np.sum(predictions, axis=0)
            
            mse = mean_squared_error(y, ensemble_pred)
            mae = mean_absolute_error(y, ensemble_pred)
            
            # Calculate directional accuracy (most important for our use case)
            y_direction = np.sign(y)
            pred_direction = np.sign(ensemble_pred)
            directional_accuracy = np.mean(y_direction == pred_direction)
            
            logger.info(f"Training Metrics - MSE: {mse:.6f}, MAE: {mae:.6f}, "
                       f"Directional Accuracy: {directional_accuracy:.3f}")
            
        except Exception as e:
            logger.error(f"Error calculating training metrics: {e}")
    
    def save_model(self):
        """Save the trained model to disk"""
        try:
            model_data = {
                'models': self.models,
                'scaler': self.scaler,
                'is_trained': self.is_trained,
                'feature_history': self.feature_history[-100:],  # Save last 100 samples
                'target_history': self.target_history[-100:],
                'model_weights': self.model_weights
            }
            
            os.makedirs('models', exist_ok=True)
            with open('models/ensemble_predictor.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info("Model saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists('models/ensemble_predictor.pkl'):
                with open('models/ensemble_predictor.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                
                self.models = model_data['models']
                self.scaler = model_data['scaler']
                self.is_trained = model_data['is_trained']
                self.feature_history = model_data.get('feature_history', [])
                self.target_history = model_data.get('target_history', [])
                self.model_weights = model_data.get('model_weights', self.model_weights)
                
                logger.info("Model loaded successfully")
            else:
                logger.info("No saved model found, starting fresh")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")

class SimpleMovingAveragePredictor:
    """Simple baseline predictor using moving averages"""
    
    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.price_history = []
    
    def add_price(self, price: float):
        """Add new price to history"""
        self.price_history.append(price)
        if len(self.price_history) > self.window_size * 2:
            self.price_history = self.price_history[-self.window_size * 2:]
    
    def predict(self) -> float:
        """Predict price direction based on moving average crossover"""
        if len(self.price_history) < self.window_size:
            return 0.0
        
        try:
            # Short and long moving averages
            short_ma = np.mean(self.price_history[-self.window_size//2:])
            long_ma = np.mean(self.price_history[-self.window_size:])
            
            # Predict based on crossover
            if short_ma > long_ma:
                return 0.01  # Bullish signal
            elif short_ma < long_ma:
                return -0.01  # Bearish signal
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error in MA prediction: {e}")
            return 0.0

class AdaptivePredictor:
    """Adaptive predictor that switches between models based on performance"""
    
    def __init__(self):
        self.ensemble = EnsemblePredictor()
        self.ma_predictor = SimpleMovingAveragePredictor()
        
        # Performance tracking
        self.ensemble_scores = []
        self.ma_scores = []
        self.recent_predictions = []
        self.recent_actuals = []
        self.performance_window = 50
        
        # Adaptive weights
        self.ensemble_weight = 0.7
        self.ma_weight = 0.3
    
    def add_training_data(self, features: np.ndarray, price: float, target: Optional[float] = None):
        """Add training data to models"""
        self.ensemble.add_training_sample(features, target if target is not None else 0.0)
        self.ma_predictor.add_price(price)
        
        # Update model performance if we have target
        if target is not None and len(self.recent_predictions) > 0:
            self._update_performance(target)
    
    def predict(self, features: np.ndarray, current_price: float) -> np.ndarray:
        """Make adaptive prediction combining multiple models"""
        try:
            # Get predictions from both models
            ensemble_pred, ensemble_conf = self.ensemble.predict(features)
            ma_pred = self.ma_predictor.predict()
            
            # Adaptive weighting based on recent performance
            if len(self.ensemble_scores) > 10 and len(self.ma_scores) > 10:
                ensemble_recent_perf = np.mean(self.ensemble_scores[-10:])
                ma_recent_perf = np.mean(self.ma_scores[-10:])
                
                # Update weights based on relative performance
                total_perf = ensemble_recent_perf + ma_recent_perf + 1e-8
                self.ensemble_weight = ensemble_recent_perf / total_perf
                self.ma_weight = ma_recent_perf / total_perf
            
            # Combine predictions
            final_pred = (ensemble_pred * self.ensemble_weight + 
                         ma_pred * self.ma_weight)
            
            # Store prediction for performance tracking
            self.recent_predictions.append(final_pred)
            if len(self.recent_predictions) > self.performance_window:
                self.recent_predictions.pop(0)
            
            # Convert to feature vector for the validator
            # The validator expects 100 features that somehow encode the prediction
            prediction_vector = self._encode_prediction_to_features(
                final_pred, ensemble_conf, current_price, features
            )
            
            logger.info(f"Adaptive prediction: {final_pred:.4f}, "
                       f"Ensemble weight: {self.ensemble_weight:.3f}")
            
            return prediction_vector
            
        except Exception as e:
            logger.error(f"Error in adaptive prediction: {e}")
            return np.zeros(config.feature_length)
    
    def _encode_prediction_to_features(self, prediction: float, confidence: float, 
                                     current_price: float, market_features: np.ndarray) -> np.ndarray:
        """Encode prediction into feature vector for the validator"""
        try:
            # Strategy: Create a feature vector that encodes our prediction
            # The validator's ML model will learn to extract this signal
            
            # Start with market features as base
            features = market_features.copy()
            
            # Encode prediction strength in multiple ways
            pred_strength = abs(prediction)
            pred_direction = np.sign(prediction)
            
            # Method 1: Amplitude modulation of existing features
            if pred_strength > 0.001:  # Only if we have a strong prediction
                # Amplify features in proportion to prediction confidence
                amplification = 1.0 + (pred_strength * confidence * 0.5)
                features[:20] *= amplification  # Amplify first 20 features
            
            # Method 2: Add prediction-specific features at the end
            if len(features) >= 90:
                # Reserve last 10 features for prediction encoding
                features[-10] = prediction * 10  # Scaled prediction
                features[-9] = confidence
                features[-8] = pred_direction
                features[-7] = pred_strength * 100  # Scaled prediction strength
                features[-6] = np.sin(prediction * np.pi)  # Non-linear encoding
                features[-5] = np.cos(prediction * np.pi)
                features[-4] = prediction ** 2 * np.sign(prediction)  # Squared with sign
                features[-3] = np.tanh(prediction * 5)  # Saturated prediction
                features[-2] = np.log(1 + abs(prediction)) * pred_direction  # Log scaling
                features[-1] = np.exp(-abs(prediction)) * pred_direction  # Exponential decay
            
            # Method 3: Phase modulation (shift features based on prediction)
            if abs(prediction) > 0.001:
                # Slightly shift feature values to encode prediction
                shift_amount = prediction * 0.1
                features[30:50] += shift_amount
            
            # Ensure features are in valid range [-1, 1]
            features = np.clip(features, -1.0, 1.0)
            
            return features
            
        except Exception as e:
            logger.error(f"Error encoding prediction to features: {e}")
            return market_features  # Fallback to original features
    
    def _update_performance(self, actual: float):
        """Update performance metrics for both models"""
        if len(self.recent_predictions) > 0:
            last_pred = self.recent_predictions[-1]
            
            # Calculate directional accuracy (most important)
            actual_direction = np.sign(actual)
            pred_direction = np.sign(last_pred)
            directional_accuracy = 1.0 if actual_direction == pred_direction else 0.0
            
            # For ensemble vs MA, we approximate their individual contributions
            ensemble_score = directional_accuracy * self.ensemble_weight
            ma_score = directional_accuracy * self.ma_weight
            
            self.ensemble_scores.append(ensemble_score)
            self.ma_scores.append(ma_score)
            
            # Keep only recent scores
            if len(self.ensemble_scores) > self.performance_window:
                self.ensemble_scores.pop(0)
            if len(self.ma_scores) > self.performance_window:
                self.ma_scores.pop(0)
    
    def retrain_if_needed(self):
        """Retrain ensemble model if we have enough new data"""
        if len(self.ensemble.feature_history) >= 50:
            if len(self.ensemble.feature_history) % 20 == 0:  # Retrain every 20 samples
                logger.info("Retraining ensemble model...")
                self.ensemble.train()

# Global predictor instance
predictor = AdaptivePredictor()