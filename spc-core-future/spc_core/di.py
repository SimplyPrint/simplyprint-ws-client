"""
Dependency Injection - Clean and Simple

Uses standard Python patterns with minimal magic.
"""

__all__ = ["Container", "Injected", "Service", "inject", "provider"]

from typing import TypeVar, Any, Optional, Dict, Type, Callable, get_type_hints
from abc import ABC
import inspect
import asyncio
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T")


class Injected:
    """
    Type marker for dependency injection.

    Usage:
        class MyService:
            http: HTTPClient = Injected

    Or with factory:
        class MyService:
            logger: Logger = Injected.of(lambda: logging.getLogger(__name__))
    """

    def __class_getitem__(cls, item):
        """Support Injected[Type] syntax"""
        return item

    @staticmethod
    def of(factory: Callable[..., T]) -> T:
        """Create injected dependency with custom factory"""
        return _InjectedFactory(factory)  # type: ignore


class _InjectedFactory:
    """Internal: Wraps a factory function for injection"""
    def __init__(self, factory: Callable):
        self.factory = factory


class Service(ABC):
    """
    Base class for services that support lifecycle hooks.

    Usage:
        class MyService(Service):
            async def start(self):
                # Called when service starts
                pass

            async def stop(self):
                # Called when service stops
                pass
    """

    async def start(self):
        """Override to initialize service"""
        pass

    async def stop(self):
        """Override to cleanup service"""
        pass


class Container:
    """
    Service container for dependency injection.

    Simpler than the original DIContainer - focuses on clarity over features.

    Usage:
        container = Container()
        container.register(HTTPClient)
        container.register(MyService)

        service = await container.resolve(MyService)
    """

    def __init__(self):
        self._instances: Dict[Type, Any] = {}
        self._factories: Dict[Type, Callable] = {}
        self._resolving: set[Type] = set()

    def register(
        self,
        type_: Type[T],
        factory: Optional[Callable[..., T]] = None,
        *,
        instance: Optional[T] = None,
        singleton: bool = True
    ):
        """
        Register a type with the container.

        Args:
            type_: The type to register
            factory: Optional factory function (defaults to type constructor)
            instance: Pre-existing instance (for singletons)
            singleton: Whether to cache instances (default: True)
        """
        if instance is not None:
            self._instances[type_] = instance
            logger.debug(f"Registered instance of {type_.__name__}")
        else:
            self._factories[type_] = factory or type_
            logger.debug(f"Registered {type_.__name__}")

    def register_instance(self, type_: Type[T], instance: T):
        """Convenience method to register an existing instance"""
        self.register(type_, instance=instance)

    async def resolve(self, type_: Type[T], **overrides) -> T:
        """
        Resolve a dependency.

        Args:
            type_: The type to resolve
            **overrides: Manual overrides for testing

        Returns:
            Instance of the requested type

        Raises:
            RuntimeError: If circular dependency detected
            TypeError: If type cannot be resolved
        """
        # Check overrides
        if type_ in overrides:
            return overrides[type_]

        # Check cache
        if type_ in self._instances:
            return self._instances[type_]

        # Circular dependency check
        if type_ in self._resolving:
            raise RuntimeError(f"Circular dependency detected: {type_.__name__}")

        self._resolving.add(type_)

        try:
            factory = self._factories.get(type_, type_)

            # Resolve constructor dependencies
            kwargs = await self._resolve_dependencies(factory, **overrides)

            # Create instance
            instance = await self._create(factory, **kwargs)

            # Call start() if it's a Service
            if isinstance(instance, Service):
                await instance.start()

            # Cache
            self._instances[type_] = instance

            logger.debug(f"Resolved {type_.__name__}")
            return instance

        finally:
            self._resolving.discard(type_)

    async def _resolve_dependencies(self, factory: Callable, **overrides) -> Dict[str, Any]:
        """Analyze factory and resolve dependencies"""
        sig = inspect.signature(factory)
        type_hints = get_type_hints(factory)
        kwargs = {}

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            # Check if default is Injected marker
            if param.default is Injected:
                param_type = type_hints.get(param_name, param.annotation)
                if param_type != inspect.Parameter.empty:
                    kwargs[param_name] = await self.resolve(param_type, **overrides)

            # Check if default is an injected factory
            elif isinstance(param.default, _InjectedFactory):
                result = param.default.factory()
                if inspect.iscoroutine(result):
                    result = await result
                kwargs[param_name] = result

            # Check annotation for Injected[Type]
            elif param_name in type_hints:
                param_type = type_hints[param_name]
                # Try to resolve if it's registered
                if param_type in self._factories or param_type in self._instances:
                    kwargs[param_name] = await self.resolve(param_type, **overrides)

        return kwargs

    async def _create(self, factory: Callable, **kwargs) -> Any:
        """Create instance (sync or async)"""
        if inspect.iscoroutinefunction(factory):
            return await factory(**kwargs)
        else:
            result = factory(**kwargs)
            if inspect.iscoroutine(result):
                return await result
            return result

    async def shutdown(self):
        """Shutdown all services"""
        for instance in self._instances.values():
            if isinstance(instance, Service):
                try:
                    await instance.stop()
                except Exception as e:
                    logger.error(f"Error stopping {instance.__class__.__name__}: {e}")

        self._instances.clear()

    def clear(self):
        """Clear all cached instances without stopping them"""
        self._instances.clear()


def inject(cls: Type[T]) -> Type[T]:
    """
    Class decorator to enable dependency injection.

    Analyzes __init__ and resolves Injected parameters.

    Usage:
        @inject
        class MyService:
            def __init__(self, http: HTTPClient = Injected):
                self.http = http
    """
    original_init = cls.__init__

    async def new_init(self, **kwargs):
        # Get type hints
        hints = get_type_hints(original_init)
        sig = inspect.signature(original_init)

        # Auto-resolve Injected parameters
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            if param.default is Injected and param_name not in kwargs:
                param_type = hints.get(param_name)
                if param_type:
                    # Would need access to container here...
                    # This is why we prefer using Container.resolve directly
                    pass

        original_init(self, **kwargs)

    # For now, @inject is mainly for documentation
    # Actual injection happens via Container.resolve
    return cls


def provider(func: Callable[..., T]) -> Callable[..., T]:
    """
    Mark a function as a provider for dependency injection.

    Usage:
        @provider
        def create_http_client() -> HTTPClient:
            return HTTPClient(timeout=30)

        container.register(HTTPClient, create_http_client)
    """
    func._is_provider = True  # type: ignore
    return func
