"""
External State Modeling

Inspired by Bambu Lab's sophisticated state tracking system.
Provides change tracking, nested model updates, and atomic fields.
"""

from __future__ import annotations

from typing import TypeVar, Generic, NamedTuple, Any, Union, TYPE_CHECKING
from pydantic import BaseModel, Field
from pydantic.fields import FieldInfo

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "Changed",
    "ChangedFields",
    "Atomic",
    "Metadata",
    "StatefulModel",
    "ExternalStateModel",
]

# ============================================================================
# Change Tracking
# ============================================================================

T = TypeVar("T")


class Changed(NamedTuple, Generic[T]):
    """
    Tracks a field change with old and new values.

    Similar to Bambu Lab's UpdatedField pattern.

    Example:
        >>> change = Changed(old=25.0, new=200.0)
        >>> change.has_changed()
        True
        >>> change.delta()
        175.0
    """
    old: T | None
    new: T | None

    def has_changed(self) -> bool:
        """Check if the value actually changed"""
        return self.old != self.new

    def delta(self) -> T | None:
        """
        Calculate numeric delta between old and new.

        Returns None if not numeric or if either value is None.
        """
        if self.old is None or self.new is None:
            return None

        try:
            return self.new - self.old  # type: ignore
        except TypeError:
            return None


# Type alias for changed fields
ChangedFields = dict[str, Union[Changed, "ChangedFields"]]


# ============================================================================
# Field Markers
# ============================================================================


def Atomic(default: Any = None, **kwargs: Any) -> FieldInfo:
    """
    Mark a field as atomic (non-recursive).

    Atomic fields are not recursively updated even if they contain models.
    The entire value is replaced on update.

    Example:
        >>> class State(ExternalStateModel):
        ...     config: dict[str, Any] = Atomic(default_factory=dict)
        ...     # config is replaced entirely, not merged

    Args:
        default: Default value for the field
        **kwargs: Additional Pydantic Field arguments (e.g., default_factory, description)

    Returns:
        Pydantic FieldInfo with atomic marker
    """
    return Field(default=default, json_schema_extra={"atomic": True}, **kwargs)


def Metadata(**kwargs: Any) -> FieldInfo:
    """
    Mark a field as metadata (excluded from updates).

    Metadata fields are never updated from external state.
    Use for internal tracking, timestamps, etc.

    Example:
        >>> class State(ExternalStateModel):
        ...     last_update: float = Metadata(default=0.0)
        ...     # last_update is never updated from external state

    Args:
        **kwargs: Pydantic Field arguments (e.g., default, default_factory, description)

    Returns:
        Pydantic FieldInfo with metadata marker
    """
    return Field(json_schema_extra={"metadata": True}, **kwargs)


def is_atomic(field_info: FieldInfo) -> bool:
    """
    Check if a field is marked as atomic.

    Args:
        field_info: Pydantic FieldInfo object

    Returns:
        True if field is atomic, False otherwise
    """
    if not hasattr(field_info, "json_schema_extra"):
        return False
    extra = field_info.json_schema_extra
    if extra is None:
        return False
    return bool(extra.get("atomic", False))


def is_metadata(field_info: FieldInfo) -> bool:
    """
    Check if a field is marked as metadata.

    Args:
        field_info: Pydantic FieldInfo object

    Returns:
        True if field is metadata, False otherwise
    """
    if not hasattr(field_info, "json_schema_extra"):
        return False
    extra = field_info.json_schema_extra
    if extra is None:
        return False
    return bool(extra.get("metadata", False))


# ============================================================================
# Stateful Model Protocol
# ============================================================================

class StatefulModel:
    """
    Protocol for models that support state updates with change tracking.

    Any model implementing this protocol can be updated recursively.
    """

    def update_from(self, other: "StatefulModel") -> ChangedFields:
        """
        Update this model from another model, returning changes.

        Args:
            other: The source model to update from

        Returns:
            Dictionary of changed fields with old/new values
        """
        raise NotImplementedError


# ============================================================================
# External State Model
# ============================================================================

class ExternalStateModel(BaseModel, StatefulModel):
    """
    Base class for external state models with smart change tracking.

    Inspired by Bambu Lab's SimpleUpdateModel. Provides:
    - Automatic change detection
    - Recursive updates for nested models
    - Atomic field support (non-recursive)
    - Metadata field exclusion

    Example:
        >>> class ToolState(ExternalStateModel):
        ...     temperature: float = 0.0
        ...     target: float = 0.0
        ...
        >>> class PrinterState(ExternalStateModel):
        ...     status: str = "idle"
        ...     tool: ToolState = Field(default_factory=ToolState)
        ...     last_update: float = Metadata(default=0.0)
        ...
        >>> state = PrinterState()
        >>> new_state = PrinterState(
        ...     status="printing",
        ...     tool=ToolState(temperature=200.0, target=200.0)
        ... )
        >>> changes = state.update_from(new_state)
        >>> print(changes)
        {
            'status': Changed(old='idle', new='printing'),
            'tool': {
                'temperature': Changed(old=0.0, new=200.0),
                'target': Changed(old=0.0, new=200.0)
            }
        }
    """

    def update_from(self, other: "ExternalStateModel") -> ChangedFields:
        """
        Update this model from another model, tracking all changes.

        Recursively updates nested StatefulModel instances unless
        marked as Atomic. Skips Metadata fields entirely.

        Args:
            other: Source model to update from (must be same type)

        Returns:
            Nested dictionary of Changed objects showing what changed

        Example:
            >>> old_state = PrinterState(status="idle")
            >>> new_state = PrinterState(status="printing")
            >>> changes = old_state.update_from(new_state)
            >>> changes["status"].has_changed()
            True
        """
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"Cannot update {self.__class__.__name__} from {other.__class__.__name__}"
            )

        changes: ChangedFields = {}

        # Iterate over all fields in the source model
        for field_name, field_info in other.model_fields.items():
            # Skip metadata fields
            if is_metadata(field_info):
                continue

            new_value = getattr(other, field_name)
            old_value = getattr(self, field_name)

            # Handle recursive updates for nested models (unless atomic)
            if not is_atomic(field_info):
                # Check if it's a nested StatefulModel
                if isinstance(old_value, StatefulModel) and isinstance(new_value, StatefulModel):
                    nested_changes = old_value.update_from(new_value)
                    if nested_changes:
                        changes[field_name] = nested_changes
                    continue

                # Handle lists of models
                if isinstance(old_value, list) and isinstance(new_value, list):
                    list_changes = self._update_list(old_value, new_value)
                    if list_changes:
                        changes[field_name] = list_changes
                    continue

                # Handle dicts of models
                if isinstance(old_value, dict) and isinstance(new_value, dict):
                    dict_changes = self._update_dict(old_value, new_value)
                    if dict_changes:
                        changes[field_name] = dict_changes
                    continue

            # Atomic update: replace entire value
            setattr(self, field_name, new_value)
            change = Changed(old=old_value, new=new_value)

            if change.has_changed():
                changes[field_name] = change

        return changes

    def _update_list(
        self, old_list: list[Any], new_list: list[Any]
    ) -> ChangedFields | None:
        """
        Update a list of models, tracking changes by index.

        If lists differ in length, the entire list is replaced atomically.

        Args:
            old_list: Current list to update in place
            new_list: New list with updated values

        Returns:
            Dictionary of changes by index, or None if no changes
        """
        if len(old_list) != len(new_list):
            # Length mismatch - atomic replacement
            return {"_list_replaced": Changed(old=old_list[:], new=new_list[:])}

        list_changes: dict[Union[int, str], Union[Changed, ChangedFields]] = {}
        for idx, (old_item, new_item) in enumerate(zip(old_list, new_list)):
            if isinstance(old_item, StatefulModel) and isinstance(new_item, StatefulModel):
                item_changes = old_item.update_from(new_item)
                if item_changes:
                    list_changes[idx] = item_changes
            elif old_item != new_item:
                old_list[idx] = new_item
                list_changes[idx] = Changed(old=old_item, new=new_item)

        return list_changes if list_changes else None

    def _update_dict(
        self, old_dict: dict[str, Any], new_dict: dict[str, Any]
    ) -> ChangedFields | None:
        """
        Update a dictionary of models, tracking changes by key.

        Handles added/removed keys and nested model updates.

        Args:
            old_dict: Current dictionary to update in place
            new_dict: New dictionary with updated values

        Returns:
            Dictionary of changes by key, or None if no changes
        """
        dict_changes: ChangedFields = {}

        # Check existing keys
        for key in old_dict:
            if key not in new_dict:
                # Key removed
                dict_changes[f"_{key}_removed"] = Changed(old=old_dict[key], new=None)
            else:
                old_val = old_dict[key]
                new_val = new_dict[key]

                if isinstance(old_val, StatefulModel) and isinstance(new_val, StatefulModel):
                    nested_changes = old_val.update_from(new_val)
                    if nested_changes:
                        dict_changes[key] = nested_changes
                elif old_val != new_val:
                    old_dict[key] = new_val
                    dict_changes[key] = Changed(old=old_val, new=new_val)

        # Check for added keys
        for key in new_dict:
            if key not in old_dict:
                old_dict[key] = new_dict[key]
                dict_changes[f"_{key}_added"] = Changed(old=None, new=new_dict[key])

        return dict_changes if dict_changes else None

    def has_changed(self, changes: ChangedFields, field_path: str) -> bool:
        """
        Check if a specific field changed.

        Supports dot notation for nested fields.

        Example:
            >>> changes = state.update_from(new_state)
            >>> state.has_changed(changes, "tool.temperature")
            True
        """
        parts = field_path.split(".", 1)
        field = parts[0]

        if field not in changes:
            return False

        change = changes[field]

        # Leaf field
        if len(parts) == 1:
            if isinstance(change, Changed):
                return change.has_changed()
            else:
                # Nested changes exist
                return True

        # Nested field
        if isinstance(change, dict):
            return self.has_changed(change, parts[1])

        return False

    def get_change(self, changes: ChangedFields, field_path: str) -> Changed | None:
        """
        Get the change object for a specific field.

        Supports dot notation for nested fields.

        Example:
            >>> changes = state.update_from(new_state)
            >>> temp_change = state.get_change(changes, "tool.temperature")
            >>> print(f"Changed from {temp_change.old} to {temp_change.new}")
        """
        parts = field_path.split(".", 1)
        field = parts[0]

        if field not in changes:
            return None

        change = changes[field]

        # Leaf field
        if len(parts) == 1:
            return change if isinstance(change, Changed) else None

        # Nested field
        if isinstance(change, dict):
            return self.get_change(change, parts[1])

        return None
