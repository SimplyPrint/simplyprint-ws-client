"""
State Mapping Utilities

Type-safe transformations between external and internal state.
Inspired by patterns in Duet3D integration.
"""

from typing import TypeVar, Generic, Callable, Any, get_type_hints, Optional, Dict, List
from functools import wraps
from pydantic import BaseModel

__all__ = [
    "StateMapper",
    "FieldMapper",
    "mapper",
    "field_mapper",
]

# ============================================================================
# Types
# ============================================================================

ExternalT = TypeVar("ExternalT", bound=BaseModel)
InternalT = TypeVar("InternalT", bound=BaseModel)


# ============================================================================
# Field Mapper
# ============================================================================

class FieldMapper(Generic[ExternalT, InternalT]):
    """
    Maps individual fields between external and internal models.

    Example:
        >>> temp_mapper = FieldMapper(
        ...     extract=lambda ext: ext.temperature.actual,
        ...     transform=lambda val: round(val, 1)
        ... )
        >>> internal_temp = temp_mapper.map(external_state)
    """

    def __init__(
        self,
        extract: Optional[Callable[[ExternalT], Any]] = None,
        transform: Optional[Callable[[Any], Any]] = None,
        default: Any = None,
    ):
        """
        Initialize field mapper.

        Args:
            extract: Function to extract value from external model
            transform: Function to transform the extracted value
            default: Default value if extraction fails
        """
        self._extract = extract
        self._transform = transform
        self._default = default

    def map(self, external: ExternalT) -> Any:
        """
        Map field from external to internal representation.

        Args:
            external: External model instance

        Returns:
            Mapped value or default if extraction fails
        """
        try:
            # Extract value
            if self._extract:
                value = self._extract(external)
            else:
                value = external

            # Transform value
            if self._transform:
                value = self._transform(value)

            return value

        except (AttributeError, KeyError, TypeError):
            return self._default


# ============================================================================
# State Mapper
# ============================================================================

class StateMapper(Generic[ExternalT, InternalT]):
    """
    Maps complete state between external and internal models.

    Provides type-safe transformation with field-level mapping.

    Example:
        >>> class ExternalState(BaseModel):
        ...     temp_actual: float
        ...     temp_target: float
        ...     status_code: int
        ...
        >>> class InternalState(BaseModel):
        ...     temperature: float
        ...     target_temperature: float
        ...     status: str
        ...
        >>> mapper = StateMapper(
        ...     internal_type=InternalState,
        ...     field_mappings={
        ...         "temperature": FieldMapper(
        ...             extract=lambda e: e.temp_actual
        ...         ),
        ...         "target_temperature": FieldMapper(
        ...             extract=lambda e: e.temp_target
        ...         ),
        ...         "status": FieldMapper(
        ...             extract=lambda e: e.status_code,
        ...             transform=lambda code: {0: "idle", 1: "printing"}.get(code, "unknown")
        ...         )
        ...     }
        ... )
        ...
        >>> external = ExternalState(temp_actual=200.0, temp_target=200.0, status_code=1)
        >>> internal = mapper.map(external)
        >>> print(internal.status)  # "printing"
    """

    def __init__(
        self,
        internal_type: type[InternalT],
        field_mappings: Optional[Dict[str, FieldMapper[ExternalT, Any]]] = None,
        strict: bool = True,
    ):
        """
        Initialize state mapper.

        Args:
            internal_type: Target internal model type
            field_mappings: Dictionary of field name -> FieldMapper
            strict: If True, require all internal fields to be mapped
        """
        self._internal_type = internal_type
        self._field_mappings = field_mappings or {}
        self._strict = strict

        if strict:
            self._validate_mappings()

    def _validate_mappings(self) -> None:
        """Validate that all required fields are mapped"""
        try:
            type_hints = get_type_hints(self._internal_type)
            internal_fields = set(type_hints.keys())
            mapped_fields = set(self._field_mappings.keys())

            missing = internal_fields - mapped_fields
            if missing:
                raise ValueError(
                    f"Missing mappings for fields: {missing}. "
                    f"Set strict=False to allow unmapped fields."
                )
        except Exception:
            # If we can't get type hints, skip validation
            pass

    def map(self, external: ExternalT) -> InternalT:
        """
        Map external state to internal state.

        Args:
            external: External model instance

        Returns:
            Internal model instance with mapped values

        Raises:
            ValueError: If strict=True and a required field cannot be mapped
        """
        mapped_values = {}

        for field_name, field_mapper in self._field_mappings.items():
            mapped_values[field_name] = field_mapper.map(external)

        try:
            return self._internal_type(**mapped_values)
        except Exception as e:
            if self._strict:
                raise ValueError(f"Failed to create {self._internal_type.__name__}: {e}") from e
            raise

    def map_partial(self, external: ExternalT, fields: List[str]) -> Dict[str, Any]:
        """
        Map only specific fields.

        Useful for incremental updates.

        Args:
            external: External model instance
            fields: List of field names to map

        Returns:
            Dictionary of mapped field values

        Example:
            >>> partial = mapper.map_partial(external, ["temperature", "status"])
            >>> internal_state.model_copy(update=partial)
        """
        return {
            field: self._field_mappings[field].map(external)
            for field in fields
            if field in self._field_mappings
        }


# ============================================================================
# Decorators
# ============================================================================

def mapper(
    external_type: type[ExternalT],
    internal_type: type[InternalT],
    strict: bool = True
):
    """
    Decorator to create a state mapper from field mapping methods.

    Methods decorated with @field_mapper will be used as field mappings.

    Example:
        >>> @mapper(ExternalState, InternalState)
        ... class MyMapper:
        ...     @field_mapper("temperature")
        ...     def map_temperature(self, external: ExternalState) -> float:
        ...         return round(external.temp_actual, 1)
        ...
        ...     @field_mapper("status")
        ...     def map_status(self, external: ExternalState) -> str:
        ...         return {0: "idle", 1: "printing"}.get(external.status_code, "unknown")
        ...
        >>> mapper_instance = MyMapper()
        >>> internal = mapper_instance.map(external_state)
    """
    def decorator(cls):
        class MappedClass(cls):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # Collect field mappings from decorated methods
                field_mappings = {}
                for attr_name in dir(self):
                    if attr_name.startswith("_"):
                        continue

                    attr = getattr(self, attr_name)
                    if hasattr(attr, "_field_mapper_name"):
                        field_name = attr._field_mapper_name
                        field_mappings[field_name] = FieldMapper(
                            extract=lambda ext, method=attr: method(ext)
                        )

                # Create state mapper
                self._state_mapper = StateMapper(
                    internal_type=internal_type,
                    field_mappings=field_mappings,
                    strict=strict
                )

            def map(self, external: external_type) -> internal_type:
                """Map external state to internal state"""
                return self._state_mapper.map(external)

            def map_partial(self, external: external_type, fields: List[str]) -> Dict[str, Any]:
                """Map only specific fields"""
                return self._state_mapper.map_partial(external, fields)

        return MappedClass

    return decorator


def field_mapper(field_name: str):
    """
    Decorator to mark a method as a field mapper.

    Used with @mapper class decorator.

    Example:
        >>> @mapper(ExternalState, InternalState)
        ... class MyMapper:
        ...     @field_mapper("temperature")
        ...     def map_temperature(self, external: ExternalState) -> float:
        ...         return external.temp_actual
    """
    def decorator(func: Callable) -> Callable:
        func._field_mapper_name = field_name
        return func

    return decorator


# ============================================================================
# Convenience Functions
# ============================================================================

def create_mapper(
    internal_type: type[InternalT],
    mappings: Dict[str, Callable[[ExternalT], Any]],
    strict: bool = True
) -> StateMapper[ExternalT, InternalT]:
    """
    Create a state mapper from a dictionary of mapping functions.

    Convenience function for simple mappings without decorators.

    Example:
        >>> mapper = create_mapper(
        ...     InternalState,
        ...     {
        ...         "temperature": lambda e: e.temp_actual,
        ...         "target_temperature": lambda e: e.temp_target,
        ...         "status": lambda e: {0: "idle", 1: "printing"}.get(e.status_code)
        ...     }
        ... )
        >>> internal = mapper.map(external)
    """
    field_mappings = {
        field_name: FieldMapper(extract=mapping_func)
        for field_name, mapping_func in mappings.items()
    }

    return StateMapper(
        internal_type=internal_type,
        field_mappings=field_mappings,
        strict=strict
    )
