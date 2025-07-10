import logging
import os
import json
from datetime import datetime
from typing import Dict, Any

class EnhancedLogger:
    """Enhanced logging system for the trading system"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger("bit_trade")
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        simple_formatter = logging.Formatter(
            '%(levelname)s: %(message)s'
        )
        
        # File handler for detailed logs
        file_handler = logging.FileHandler(f"{log_dir}/bit_trade.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Performance logger
        self.perf_logger = logging.getLogger("bit_trade.performance")
        self.perf_logger.setLevel(logging.INFO)
        
        perf_handler = logging.FileHandler(f"{log_dir}/performance.log")
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(detailed_formatter)
        self.perf_logger.addHandler(perf_handler)
        
        # Error logger
        self.error_logger = logging.getLogger("bit_trade.errors")
        self.error_logger.setLevel(logging.ERROR)
        
        error_handler = logging.FileHandler(f"{log_dir}/errors.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.error_logger.addHandler(error_handler)
    
    def info(self, message: str, extra_data: Dict = None):
        """Log info message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        self.logger.info(message)
    
    def error(self, message: str, exception: Exception = None, extra_data: Dict = None):
        """Log error message"""
        if exception:
            message = f"{message} | Exception: {str(exception)}"
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        self.logger.error(message)
        self.error_logger.error(message)
    
    def debug(self, message: str, extra_data: Dict = None):
        """Log debug message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        self.logger.debug(message)
    
    def warning(self, message: str, extra_data: Dict = None):
        """Log warning message"""
        if extra_data:
            message = f"{message} | Data: {json.dumps(extra_data, default=str)}"
        self.logger.warning(message)
    
    def performance(self, event: str, metrics: Dict[str, Any]):
        """Log performance metrics"""
        perf_data = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "metrics": metrics
        }
        self.perf_logger.info(json.dumps(perf_data, default=str))
    
    def strategy_generated(self, model: str, strategy_type: str, success: bool, metrics: Dict = None):
        """Log strategy generation event"""
        self.performance("strategy_generated", {
            "model": model,
            "strategy_type": strategy_type,
            "success": success,
            "metrics": metrics or {}
        })
    
    def model_performance(self, model: str, performance_data: Dict):
        """Log model performance update"""
        self.performance("model_performance_update", {
            "model": model,
            "performance": performance_data
        })
    
    def system_event(self, event: str, details: Dict = None):
        """Log system events"""
        self.info(f"System Event: {event}", details)
    
    def api_call(self, service: str, endpoint: str, success: bool, response_time: float = None, error: str = None):
        """Log API calls"""
        api_data = {
            "service": service,
            "endpoint": endpoint,
            "success": success,
            "response_time": response_time,
            "error": error
        }
        
        if success:
            self.info(f"API Call: {service}/{endpoint} - Success", api_data)
        else:
            self.error(f"API Call: {service}/{endpoint} - Failed", extra_data=api_data)

# Global logger instance
logger = EnhancedLogger()