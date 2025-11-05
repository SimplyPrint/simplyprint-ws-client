"""
Dependency Injection System

Provides FastAPI-style dependency injection with:
- Type-based resolution
- Recursive dependency graph
- Singleton and transient scopes
- Async support
"""

__all__ = ["Depends", "DIContainer", "Injectable", "get_container"]

from typing import TypeVar, Generic, Callable, Any, Optional, Dict, Set, Type
from dataclasses import dataclass
import inspect
import asyncio
import logging
from abc import ABC, abstractmethod

T = TypeVar("T")

logger = logging.getLogger(__name__)


@dataclass
class DependencyMarker(Generic[T]):
    """
    Marker class for dependency injection.
    Not meant to be instantiated directly - use Depends() function.
    """

    factory: Optional[Callable[..., T]] = None
    scope: str = "singleton"  # "singleton" | "transient"
    use_cache: bool = True

    def __repr__(self):
        return f"Depends({self.factory.__name__ if self.factory else 'inferred'}, scope={self.scope})"


def Depends(
    dependency: Optional[Callable[..., T]] = None,
    *,
    scope: str = "singleton",
    use_cache: bool = True,
) -> T:
    """
    Mark a parameter as a dependency to be injected.

    Args:
        dependency: Optional factory function. If None, inferred from type annotation.
        scope: "singleton" (default) or "transient"
        use_cache: Whether to cache singleton instances

    Usage:
        ```python
        class MyService:
            http: HTTPClient = Depends()
            config: MyConfig = Depends(MyConfig)
            logger: Logger = Depends(lambda: logging.getLogger(__name__), scope="transient")
        ```

    Note: This function returns a DependencyMarker, which is resolved by DIContainer.
    The return type annotation T is for IDE type checking only.
    """
    # Return a marker that will be resolved by DIContainer
    return DependencyMarker(factory=dependency, scope=scope, use_cache=use_cache)  # type: ignore


class Injectable(ABC):
    """
    Optional base class for injectable components.
    Provides lifecycle hooks.
    """

    async def __post_init__(self):
        """Called after dependency injection is complete"""
        pass

    async def __pre_destroy__(self):
        """Called before component is destroyed"""
        pass


class DIContainer:
    """
    Dependency injection container.

    Responsibilities:
    - Resolve dependencies based on type annotations
    - Manage singleton instances
    - Detect circular dependencies
    - Support both sync and async factories
    """

    def __init__(self):
        self._singletons: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._resolving: Set[Type] = set()  # Circular dependency detection
        self._lock = asyncio.Lock()

    def register(self, type_: Type, factory: Optional[Callable] = None, *, singleton: bool = True):
        """
        Register a type with its factory function.

        Args:
            type_: The type to register
            factory: Factory function (defaults to type itself)
            singleton: Whether to cache instances
        """
        self._factories[type_] = factory or type_
        logger.debug(f"Registered {type_.__name__} with factory {factory}")

    def register_instance(self, type_: Type, instance: Any):
        """Register a pre-existing instance as a singleton"""
        self._singletons[type_] = instance
        logger.debug(f"Registered instance of {type_.__name__}")

    async def resolve(self, type_: Type, **overrides) -> Any:
        """
        Resolve a dependency asynchronously.

        Args:
            type_: The type to resolve
            overrides: Manual overrides for specific types (useful for testing)

        Returns:
            Resolved instance

        Raises:
            RuntimeError: If circular dependency detected
            TypeError: If unable to resolve type
        """
        # Check overrides first (useful for testing)
        if type_ in overrides:
            logger.debug(f"Using override for {type_.__name__}")
            return overrides[type_]

        # Check singleton cache
        if type_ in self._singletons:
            logger.debug(f"Returning cached singleton for {type_.__name__}")
            return self._singletons[type_]

        # Prevent concurrent resolution of the same type
        async with self._lock:
            # Double-check cache after acquiring lock
            if type_ in self._singletons:
                return self._singletons[type_]

            # Circular dependency detection
            if type_ in self._resolving:
                chain = " -> ".join(t.__name__ for t in self._resolving)
                raise RuntimeError(
                    f"Circular dependency detected: {chain} -> {type_.__name__}"
                )

            self._resolving.add(type_)

            try:
                # Get factory (either registered or the type itself)
                factory = self._factories.get(type_, type_)

                # Build dependencies for this factory
                kwargs = await self._resolve_dependencies(factory, **overrides)

                # Create instance
                instance = await self._create_instance(factory, **kwargs)

                # Call post-init hook if available
                if isinstance(instance, Injectable):
                    await instance.__post_init__()

                # Cache as singleton
                self._singletons[type_] = instance

                logger.debug(f"Created instance of {type_.__name__}")
                return instance

            finally:
                self._resolving.discard(type_)

    async def _resolve_dependencies(self, factory: Callable, **overrides) -> Dict[str, Any]:
        """
        Analyze factory signature and resolve all dependencies.

        Args:
            factory: The factory function to analyze
            overrides: Manual overrides

        Returns:
            Dictionary of resolved dependencies keyed by parameter name
        """
        sig = inspect.signature(factory)
        kwargs = {}

        for param_name, param in sig.parameters.items():
            # Skip self/cls
            if param_name in ("self", "cls"):
                continue

            # Skip if no type annotation
            if param.annotation == inspect.Parameter.empty:
                continue

            # Check if default value is a DependencyMarker
            if isinstance(param.default, DependencyMarker):
                marker: DependencyMarker = param.default

                # Determine what to resolve
                dep_type = marker.factory or param.annotation
                if not isinstance(dep_type, type):
                    # It's a factory function, call it
                    dep_type = marker.factory

                # Resolve based on scope
                if marker.scope == "singleton" and marker.use_cache:
                    kwargs[param_name] = await self.resolve(dep_type, **overrides)
                else:
                    # Transient: create new instance every time
                    kwargs[param_name] = await self._create_instance(dep_type)

            # If no default and parameter is required, try to resolve by type
            elif param.default == inspect.Parameter.empty:
                try:
                    kwargs[param_name] = await self.resolve(param.annotation, **overrides)
                except Exception as e:
                    logger.warning(
                        f"Could not resolve required parameter '{param_name}' "
                        f"of type {param.annotation}: {e}"
                    )
                    # Let it fail when calling the factory
                    pass

        return kwargs

    async def _create_instance(self, factory: Callable, **kwargs) -> Any:
        """
        Create an instance using the factory, handling both sync and async.

        Args:
            factory: The factory function
            kwargs: Arguments to pass to factory

        Returns:
            Created instance
        """
        if inspect.iscoroutinefunction(factory):
            # Async factory
            return await factory(**kwargs)
        else:
            # Sync factory
            result = factory(**kwargs)

            # Check if result is a coroutine (async __init__)
            if inspect.iscoroutine(result):
                return await result

            return result

    def clear(self):
        """Clear all cached singletons"""
        self._singletons.clear()
        logger.debug("Cleared all singletons")

    async def destroy_all(self):
        """
        Call __pre_destroy__ on all singletons that support it.
        Useful for cleanup on shutdown.
        """
        for instance in self._singletons.values():
            if isinstance(instance, Injectable):
                try:
                    await instance.__pre_destroy__()
                except Exception as e:
                    logger.error(f"Error destroying {instance.__class__.__name__}: {e}")

        self._singletons.clear()


# Global container instance
_global_container: Optional[DIContainer] = None


def get_container() -> DIContainer:
    """Get the global DI container (lazy initialization)"""
    global _global_container
    if _global_container is None:
        _global_container = DIContainer()
    return _global_container


def set_container(container: DIContainer):
    """Set a custom DI container (useful for testing)"""
    global _global_container
    _global_container = container
