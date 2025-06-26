# MIT License - MANTIS Miner Monitoring and Metrics

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import threading
from datetime import datetime, timedelta

@dataclass
class MinerMetrics:
    """Metrics for tracking miner performance"""
    predictions_made: int = 0
    successful_uploads: int = 0
    failed_uploads: int = 0
    network_errors: int = 0
    prediction_accuracy: float = 0.0
    avg_prediction_confidence: float = 0.0
    uptime_seconds: float = 0.0
    last_successful_upload: Optional[float] = None
    last_error_time: Optional[float] = None
    last_error_msg: str = ""
    total_rewards_earned: float = 0.0
    current_weight: float = 0.0

class MinerMonitor:
    """Monitor and track miner performance"""
    
    def __init__(self, log_dir: str = "logs"):
        self.metrics = MinerMetrics()
        self.start_time = time.time()
        self.log_dir = log_dir
        self.metrics_file = os.path.join(log_dir, "miner_metrics.json")
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup detailed logging
        self.setup_logging()
        
        # Performance tracking
        self.recent_predictions = []
        self.recent_actuals = []
        self.performance_window = 100
        
        # Threading lock for thread-safe updates
        self.lock = threading.Lock()
        
        # Load existing metrics if available
        self.load_metrics()
        
        # Start background metrics saver
        self.start_metrics_saver()
    
    def setup_logging(self):
        """Setup comprehensive logging"""
        # Main logger
        self.logger = logging.getLogger("miner")
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # File handlers
        main_handler = logging.FileHandler(
            os.path.join(self.log_dir, "miner.log"), mode='a'
        )
        error_handler = logging.FileHandler(
            os.path.join(self.log_dir, "errors.log"), mode='a'
        )
        performance_handler = logging.FileHandler(
            os.path.join(self.log_dir, "performance.log"), mode='a'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s [%(levelname)s] - %(message)s'
        )
        
        # Configure handlers
        main_handler.setFormatter(detailed_formatter)
        error_handler.setFormatter(detailed_formatter)
        error_handler.setLevel(logging.ERROR)
        performance_handler.setFormatter(simple_formatter)
        console_handler.setFormatter(simple_formatter)
        console_handler.setLevel(logging.INFO)
        
        # Add handlers
        self.logger.addHandler(main_handler)
        self.logger.addHandler(error_handler)
        self.logger.addHandler(console_handler)
        
        # Performance logger (separate)
        self.perf_logger = logging.getLogger("performance")
        self.perf_logger.setLevel(logging.INFO)
        self.perf_logger.addHandler(performance_handler)
        
        self.logger.info("Miner monitoring initialized")
    
    def record_prediction(self, prediction: float, confidence: float):
        """Record a new prediction"""
        with self.lock:
            self.metrics.predictions_made += 1
            
            # Update average confidence
            if self.metrics.predictions_made == 1:
                self.metrics.avg_prediction_confidence = confidence
            else:
                # Running average
                alpha = 0.1  # Learning rate for running average
                self.metrics.avg_prediction_confidence = (
                    (1 - alpha) * self.metrics.avg_prediction_confidence + 
                    alpha * confidence
                )
            
            self.recent_predictions.append({
                'prediction': prediction,
                'confidence': confidence,
                'timestamp': time.time()
            })
            
            # Keep only recent predictions
            if len(self.recent_predictions) > self.performance_window:
                self.recent_predictions.pop(0)
        
        self.logger.info(f"Prediction recorded: {prediction:.4f} (conf: {confidence:.3f})")
    
    def record_upload_success(self):
        """Record successful upload"""
        with self.lock:
            self.metrics.successful_uploads += 1
            self.metrics.last_successful_upload = time.time()
        
        self.logger.info("Upload successful")
    
    def record_upload_failure(self, error_msg: str):
        """Record upload failure"""
        with self.lock:
            self.metrics.failed_uploads += 1
            self.metrics.last_error_time = time.time()
            self.metrics.last_error_msg = error_msg
        
        self.logger.error(f"Upload failed: {error_msg}")
    
    def record_network_error(self, error_msg: str):
        """Record network error"""
        with self.lock:
            self.metrics.network_errors += 1
            self.metrics.last_error_time = time.time()
            self.metrics.last_error_msg = error_msg
        
        self.logger.error(f"Network error: {error_msg}")
    
    def record_actual_outcome(self, actual_change: float):
        """Record actual price change for accuracy calculation"""
        with self.lock:
            self.recent_actuals.append({
                'actual': actual_change,
                'timestamp': time.time()
            })
            
            # Keep only recent actuals
            if len(self.recent_actuals) > self.performance_window:
                self.recent_actuals.pop(0)
            
            # Calculate accuracy if we have matching predictions
            self._update_accuracy()
    
    def _update_accuracy(self):
        """Update prediction accuracy based on recent data"""
        if len(self.recent_predictions) == 0 or len(self.recent_actuals) == 0:
            return
        
        # Match predictions with actuals based on timestamp
        matched_pairs = []
        
        for pred in self.recent_predictions:
            # Find actual outcome that occurred after this prediction
            for actual in self.recent_actuals:
                if actual['timestamp'] > pred['timestamp']:
                    time_diff = actual['timestamp'] - pred['timestamp']
                    # Only consider if actual came within reasonable time window
                    if 300 <= time_diff <= 900:  # 5-15 minutes window
                        matched_pairs.append({
                            'predicted': pred['prediction'],
                            'actual': actual['actual'],
                            'time_diff': time_diff
                        })
                    break
        
        if matched_pairs:
            # Calculate directional accuracy
            correct_directions = 0
            for pair in matched_pairs:
                pred_direction = 1 if pair['predicted'] > 0 else -1 if pair['predicted'] < 0 else 0
                actual_direction = 1 if pair['actual'] > 0 else -1 if pair['actual'] < 0 else 0
                
                if pred_direction == actual_direction:
                    correct_directions += 1
            
            self.metrics.prediction_accuracy = correct_directions / len(matched_pairs)
            
            self.perf_logger.info(
                f"Accuracy updated: {self.metrics.prediction_accuracy:.3f} "
                f"({correct_directions}/{len(matched_pairs)} correct)"
            )
    
    def update_network_weight(self, weight: float):
        """Update current network weight"""
        with self.lock:
            self.metrics.current_weight = weight
        
        self.logger.info(f"Network weight updated: {weight:.6f}")
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics snapshot"""
        with self.lock:
            # Update uptime
            self.metrics.uptime_seconds = time.time() - self.start_time
            
            # Calculate success rate
            total_uploads = self.metrics.successful_uploads + self.metrics.failed_uploads
            success_rate = (self.metrics.successful_uploads / total_uploads) if total_uploads > 0 else 0.0
            
            metrics_dict = asdict(self.metrics)
            metrics_dict['upload_success_rate'] = success_rate
            metrics_dict['uptime_hours'] = self.metrics.uptime_seconds / 3600
            
            return metrics_dict
    
    def log_status_summary(self):
        """Log comprehensive status summary"""
        metrics = self.get_current_metrics()
        
        summary = f"""
=== MINER STATUS SUMMARY ===
Uptime: {metrics['uptime_hours']:.1f} hours
Predictions Made: {metrics['predictions_made']}
Upload Success Rate: {metrics['upload_success_rate']:.1%}
Prediction Accuracy: {metrics['prediction_accuracy']:.1%}
Avg Confidence: {metrics['avg_prediction_confidence']:.3f}
Current Weight: {metrics['current_weight']:.6f}
Network Errors: {metrics['network_errors']}
Last Error: {metrics['last_error_msg'][:50]}...
===========================
        """
        
        self.logger.info(summary)
        
        # Also log recent prediction performance
        if len(self.recent_predictions) > 5:
            recent_preds = [p['prediction'] for p in self.recent_predictions[-5:]]
            recent_conf = [p['confidence'] for p in self.recent_predictions[-5:]]
            
            self.perf_logger.info(
                f"Recent predictions: {recent_preds}, "
                f"Recent confidence: {recent_conf}"
            )
    
    def save_metrics(self):
        """Save metrics to file"""
        try:
            metrics = self.get_current_metrics()
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save metrics: {e}")
    
    def load_metrics(self):
        """Load metrics from file"""
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    saved_metrics = json.load(f)
                
                # Restore relevant metrics
                self.metrics.total_rewards_earned = saved_metrics.get('total_rewards_earned', 0.0)
                self.metrics.predictions_made = saved_metrics.get('predictions_made', 0)
                self.metrics.successful_uploads = saved_metrics.get('successful_uploads', 0)
                self.metrics.failed_uploads = saved_metrics.get('failed_uploads', 0)
                
                self.logger.info("Metrics loaded from file")
        except Exception as e:
            self.logger.error(f"Failed to load metrics: {e}")
    
    def start_metrics_saver(self):
        """Start background thread to save metrics periodically"""
        def save_periodically():
            while True:
                time.sleep(300)  # Save every 5 minutes
                self.save_metrics()
        
        saver_thread = threading.Thread(target=save_periodically, daemon=True)
        saver_thread.start()
    
    def check_health(self) -> Dict[str, bool]:
        """Check miner health status"""
        current_time = time.time()
        health = {
            'predictions_active': len(self.recent_predictions) > 0,
            'uploads_working': False,
            'no_recent_errors': True,
            'reasonable_accuracy': True
        }
        
        # Check if uploads are working
        if self.metrics.last_successful_upload:
            time_since_upload = current_time - self.metrics.last_successful_upload
            health['uploads_working'] = time_since_upload < 300  # Less than 5 minutes ago
        
        # Check for recent errors
        if self.metrics.last_error_time:
            time_since_error = current_time - self.metrics.last_error_time
            health['no_recent_errors'] = time_since_error > 600  # No errors in last 10 minutes
        
        # Check prediction accuracy
        health['reasonable_accuracy'] = self.metrics.prediction_accuracy >= 0.3  # At least 30%
        
        # Log health issues
        for check, status in health.items():
            if not status:
                self.logger.warning(f"Health check failed: {check}")
        
        return health

# Global monitor instance
monitor = MinerMonitor()