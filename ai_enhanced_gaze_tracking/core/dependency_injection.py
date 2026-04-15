"""
Dependency Injection Container for flexible component management.

This module provides a simple but effective dependency injection system
that allows for easy component swapping and testing.
"""

from typing import Dict, Type, Any, Optional, Callable
import inspect
from .interfaces import (
    FaceDetector, GazeEstimator, HeadPoseEstimator, CameraCalibrator,
    SensorFusion, FocusDetector, QualityAssessor, ObjectDetector,
    CalibrationSystem
)


class DIContainer:
    """
    Dependency Injection Container for managing component instances.
    
    Supports:
    - Singleton and transient lifetimes
    - Factory functions
    - Automatic dependency resolution
    - Interface-based registration
    """
    
    def __init__(self):
        self._services: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._singletons: Dict[Type, Any] = {}
        self._configurations: Dict[str, Any] = {}
    
    def register_singleton(self, interface: Type, implementation: Type) -> None:
        """
        Register a singleton service implementation.
        
        Args:
            interface: The interface type to register
            implementation: The concrete implementation class
        """
        self._services[interface] = implementation
        self._singletons[interface] = None
    
    def register_transient(self, interface: Type, implementation: Type) -> None:
        """
        Register a transient service implementation.
        
        Args:
            interface: The interface type to register  
            implementation: The concrete implementation class
        """
        self._services[interface] = implementation
    
    def register_factory(self, interface: Type, factory: Callable) -> None:
        """
        Register a factory function for creating service instances.
        
        Args:
            interface: The interface type to register
            factory: Factory function that returns an instance
        """
        self._factories[interface] = factory
    
    def register_instance(self, interface: Type, instance: Any) -> None:
        """
        Register a pre-created instance.
        
        Args:
            interface: The interface type to register
            instance: The pre-created instance
        """
        self._singletons[interface] = instance
    
    def register_configuration(self, key: str, value: Any) -> None:
        """
        Register configuration values.
        
        Args:
            key: Configuration key
            value: Configuration value
        """
        self._configurations[key] = value
    
    def get_configuration(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        return self._configurations.get(key, default)
    
    def resolve(self, interface: Type) -> Any:
        """
        Resolve a service instance by interface type.
        
        Args:
            interface: The interface type to resolve
            
        Returns:
            Service instance
            
        Raises:
            ValueError: If interface is not registered
        """
        # Check for pre-created singleton
        if interface in self._singletons and self._singletons[interface] is not None:
            return self._singletons[interface]
        
        # Check for factory function
        if interface in self._factories:
            instance = self._factories[interface]()
            if interface in self._singletons:
                self._singletons[interface] = instance
            return instance
        
        # Check for registered service
        if interface in self._services:
            implementation = self._services[interface]
            instance = self._create_instance(implementation)
            
            # Store singleton if registered as such
            if interface in self._singletons:
                self._singletons[interface] = instance
            
            return instance
        
        raise ValueError(f"No registration found for interface: {interface}")
    
    def _create_instance(self, implementation: Type) -> Any:
        """
        Create an instance with automatic dependency injection.
        
        Args:
            implementation: The implementation class to instantiate
            
        Returns:
            Created instance with dependencies injected
        """
        # Get constructor signature
        signature = inspect.signature(implementation.__init__)
        parameters = signature.parameters
        
        # Skip 'self' parameter
        param_names = [name for name in parameters.keys() if name != 'self']
        
        # Resolve dependencies
        kwargs = {}
        for param_name in param_names:
            param = parameters[param_name]
            
            # Try to resolve by type annotation
            if param.annotation != inspect.Parameter.empty:
                try:
                    kwargs[param_name] = self.resolve(param.annotation)
                except ValueError:
                    # Check if parameter has default value
                    if param.default != inspect.Parameter.empty:
                        kwargs[param_name] = param.default
                    else:
                        # Try to resolve by parameter name from configuration
                        config_value = self.get_configuration(param_name)
                        if config_value is not None:
                            kwargs[param_name] = config_value
                        else:
                            raise ValueError(
                                f"Cannot resolve dependency '{param_name}' "
                                f"for {implementation.__name__}"
                            )
        
        return implementation(**kwargs)
    
    def clear(self) -> None:
        """Clear all registrations and cached instances."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()
        self._configurations.clear()
    
    def is_registered(self, interface: Type) -> bool:
        """
        Check if an interface is registered.
        
        Args:
            interface: The interface type to check
            
        Returns:
            True if registered, False otherwise
        """
        return (interface in self._services or 
                interface in self._factories or 
                interface in self._singletons)


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global dependency injection container."""
    return _container


def configure_default_services() -> None:
    """Configure default service registrations."""
    container = get_container()
    
    # Configuration defaults
    container.register_configuration("face_detection_confidence", 0.5)
    container.register_configuration("gaze_estimation_method", "multi_modal")
    container.register_configuration("head_pose_compensation", True)
    container.register_configuration("camera_angle_correction", True)
    container.register_configuration("ai_model_ensemble", True)
    container.register_configuration("real_time_processing", True)
    container.register_configuration("quality_assessment", True)
    container.register_configuration("focus_detection_enabled", True)
    container.register_configuration("wandering_detection_enabled", True)
    container.register_configuration("fatigue_detection_enabled", True)


# Decorator for dependency injection
def inject(interface: Type):
    """
    Decorator for automatic dependency injection.
    
    Args:
        interface: The interface type to inject
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if interface.__name__.lower() not in kwargs:
                kwargs[interface.__name__.lower()] = get_container().resolve(interface)
            return func(*args, **kwargs)
        return wrapper
    return decorator