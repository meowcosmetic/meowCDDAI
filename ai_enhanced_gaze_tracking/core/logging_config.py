"""
Logging and monitoring infrastructure for the AI-Enhanced Gaze Tracking System.

This module provides structured logging, performance monitoring, and
diagnostic capabilities throughout the system.
"""

import logging
import logging.handlers
import sys
import time
import functools
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime


class PerformanceMonitor:
    """Performance monitoring and metrics collection."""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = {}
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.perf_counter()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            return 0.0
        
        duration = time.perf_counter() - self.start_times[operation]
        
        # Update metrics
        if operation not in self.metrics:
            self.metrics[operation] = {
                'count': 0,
                'total_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0,
                'avg_time': 0.0
            }
        
        metrics = self.metrics[operation]
        metrics['count'] += 1
        metrics['total_time'] += duration
        metrics['min_time'] = min(metrics['min_time'], duration)
        metrics['max_time'] = max(metrics['max_time'], duration)
        metrics['avg_time'] = metrics['total_time'] / metrics['count']
        
        del self.start_times[operation]
        return duration
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.metrics.copy()
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.metrics.clear()
        self.start_times.clear()


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, 'component'):
            log_data['component'] = record.component
        if hasattr(record, 'operation'):
            log_data['operation'] = record.operation
        if hasattr(record, 'duration'):
            log_data['duration_ms'] = record.duration * 1000
        if hasattr(record, 'confidence'):
            log_data['confidence'] = record.confidence
        if hasattr(record, 'frame_id'):
            log_data['frame_id'] = record.frame_id
        
        return json.dumps(log_data)


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up logging configuration for the gaze tracking system.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        structured: Whether to use structured JSON logging
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('ai_enhanced_gaze_tracking')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    if structured:
        console_formatter = StructuredFormatter()
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        if structured:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific component."""
    return logging.getLogger(f'ai_enhanced_gaze_tracking.{name}')


# Global performance monitor
_performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return _performance_monitor


def log_performance(operation: str):
    """
    Decorator for automatic performance logging.
    
    Args:
        operation: Name of the operation being timed
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            monitor = get_performance_monitor()
            
            monitor.start_timer(operation)
            try:
                result = func(*args, **kwargs)
                duration = monitor.end_timer(operation)
                
                logger.info(
                    f"Operation completed: {operation}",
                    extra={
                        'component': func.__module__,
                        'operation': operation,
                        'duration': duration
                    }
                )
                
                return result
            except Exception as e:
                duration = monitor.end_timer(operation)
                logger.error(
                    f"Operation failed: {operation} - {str(e)}",
                    extra={
                        'component': func.__module__,
                        'operation': operation,
                        'duration': duration,
                        'error': str(e)
                    }
                )
                raise
        
        return wrapper
    return decorator


def log_gaze_estimate(gaze_estimate, frame_id: int = None):
    """
    Log gaze estimation results.
    
    Args:
        gaze_estimate: GazeEstimate instance
        frame_id: Optional frame identifier
    """
    logger = get_logger('gaze_estimation')
    
    extra_data = {
        'component': 'gaze_estimation',
        'confidence': gaze_estimate.confidence,
        'method': gaze_estimate.method
    }
    
    if frame_id is not None:
        extra_data['frame_id'] = frame_id
    
    logger.debug(
        f"Gaze estimated: confidence={gaze_estimate.confidence:.3f}, "
        f"method={gaze_estimate.method}",
        extra=extra_data
    )


def log_focus_event(focus_event, frame_id: int = None):
    """
    Log focus detection events.
    
    Args:
        focus_event: FocusEvent instance
        frame_id: Optional frame identifier
    """
    logger = get_logger('focus_detection')
    
    extra_data = {
        'component': 'focus_detection',
        'focus_type': focus_event.focus_type.value,
        'confidence': focus_event.confidence,
        'duration': focus_event.duration
    }
    
    if frame_id is not None:
        extra_data['frame_id'] = frame_id
    
    logger.info(
        f"Focus detected: type={focus_event.focus_type.value}, "
        f"duration={focus_event.duration:.2f}s, "
        f"confidence={focus_event.confidence:.3f}",
        extra=extra_data
    )


def log_quality_metrics(quality_metrics, frame_id: int = None):
    """
    Log data quality metrics.
    
    Args:
        quality_metrics: QualityMetrics instance
        frame_id: Optional frame identifier
    """
    logger = get_logger('quality_assessment')
    
    extra_data = {
        'component': 'quality_assessment',
        'overall_quality': quality_metrics.overall_quality,
        'head_pose_quality': quality_metrics.head_pose_quality,
        'lighting_quality': quality_metrics.lighting_quality
    }
    
    if frame_id is not None:
        extra_data['frame_id'] = frame_id
    
    if quality_metrics.overall_quality < 0.5:
        logger.warning(
            f"Low quality data detected: overall={quality_metrics.overall_quality:.3f}",
            extra=extra_data
        )
    else:
        logger.debug(
            f"Quality assessment: overall={quality_metrics.overall_quality:.3f}",
            extra=extra_data
        )