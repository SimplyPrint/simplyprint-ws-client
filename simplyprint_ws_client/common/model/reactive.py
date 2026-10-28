__all__ = ["ReactiveModel"]

import collections.abc
import datetime
import decimal
import enum
import functools
import weakref
from itertools import repeat
from typing import (
    Callable,
    Optional,
    Dict,
    Any,
    Iterable,
    Protocol,
    TypeVar,
    TYPE_CHECKING,
    ClassVar,
    Type,
    get_origin,
    Set,
    get_args,
    Union,
    no_type_check,
    cast,
    Tuple,
    List,
)

from pydantic import BaseModel, PrivateAttr
from pydantic.fields import FieldInfo

from simplyprint_ws_client.common.utils.synchronized import Synchronized
from simplyprint_ws_client.common.model.annotations import _is_exclusive, _is_untracked

TReactiveModel = TypeVar("TReactiveModel", bound="ReactiveModel")


class ReactiveContext(Protocol):
    """The contract a model's ``ctx`` callable must resolve to (or ``None``).

    The tree root's context (in practice the SimplyPrint client) stamps each
    change with a message id and signals the consumer that a changeset is ready.
    It is told *which* model and field changed, and which models were bound to
    it, so a consumer can look only where something changed instead of walking
    the whole tree for every changeset.
    """

    def stamp_change(self, model: "ReactiveModel", field: str) -> int:
        """A tracked change to ``model.field``: its id (increasing)."""
        ...

    def state_attached(self, model: "ReactiveModel") -> None:
        """``model``'s subtree was bound to this context by ``provide_context``.

        Changes it already carries are now part of the tree.
        """
        ...

    def signal(self) -> None: ...


#: A model's context accessor: a ``weakref.ref`` (or plain callable) yielding
#: the live :class:`ReactiveContext`, or ``None`` when unbound.
ContextRef = Callable[[], Optional[ReactiveContext]]


class ReactiveModel(BaseModel, Synchronized):
    """Change-tracking pydantic model: the shared reactive base.

    Adapted from pydantic-changetracker (ChangeDetectionMixin).
    MIT License
    Copyright (c) TEAM23 GmbH 2022.

    Mutations are tracked per field (an incrementing change id) and rolled up into
    a recursive changeset, which the SimplyPrint client drains into outbound
    messages. ``PrinterState`` and every nested state model build on this; the
    brand merge/diff counterpart lives in :mod:`.diff`.
    """

    COMPARABLE_TYPES: ClassVar[tuple] = (
        str,
        int,
        float,
        bool,
        enum.Enum,
        decimal.Decimal,
        datetime.datetime,
        datetime.date,
        datetime.time,
        datetime.timedelta,
        BaseModel,
    )

    @classmethod
    @functools.lru_cache(maxsize=None)
    def is_pydantic_change_detect_annotation(
        cls,
        annotation: Optional[Type[Any]] = None,
        field_info: Optional[FieldInfo] = None,
    ) -> bool:
        """
        Return True if the given annotation is a ChangeDetectionMixin annotation.
        """
        if field_info:
            if _is_untracked(field_info):
                return False

            annotation = field_info.annotation

        # if annotation is an ChangeDetectionMixin everything is easy
        if (
            get_origin(annotation)
            is None  # If the origin is None, it's likely a concrete class
            and isinstance(annotation, type)
            and issubclass(annotation, ReactiveModel)
        ):
            return True

        # Otherwise we may need to handle typing arguments. get_origin()
        # normalizes typing aliases (List, Mapping, ...) to their runtime
        # classes, so we compare against those.
        origin = get_origin(annotation)
        if origin in (list, set, frozenset, tuple):
            return cls.is_pydantic_change_detect_annotation(get_args(annotation)[0])
        elif origin is dict or (
            isinstance(origin, type) and issubclass(origin, collections.abc.Mapping)
        ):
            return cls.is_pydantic_change_detect_annotation(get_args(annotation)[1])
        elif origin is Union:
            # Note: This includes Optional, as Optional[...] is just Union[..., None]
            return any(
                cls.is_pydantic_change_detect_annotation(arg)
                for arg in get_args(annotation)
            )

        # If we did not detect an annotation, return False
        return False

    if TYPE_CHECKING:
        model_self_changed_fields: Dict[str, int] = PrivateAttr(...)
        ctx: ContextRef = PrivateAttr(...)

    __slots__ = ("model_self_changed_fields", "ctx")

    def __init__(self, ctx: ContextRef = lambda: None, **kwargs: Any) -> None:
        BaseModel.__init__(self, **kwargs)
        Synchronized.__init__(self)
        # Default to static no-context to prevent unnecessary errors
        object.__setattr__(self, "ctx", ctx)
        self.model_reset_changed()

    @classmethod
    def model_fresh(
        cls: Type[TReactiveModel], ctx: ContextRef, values: Dict[str, Any]
    ) -> TReactiveModel:
        """An instance with ``values`` (by field name, already of the field's
        type) as ``cls(ctx)`` with each then written leaves it, fields set
        included, without validation: the class's defaults are worked out once
        (its template) and copied for each instance. For a writer that
        creates many small entries (a printer's peripherals). A class with
        validators, its own post-init, private attributes, extra fields or a
        default that isn't a plain value, or ``values`` missing a required
        field, is built the validated way."""
        template = _FRESH.get(cls)
        if template is None or template[0] is not cls.__pydantic_fields__:
            template = _fresh_template(cls)
        _, defaults, required, every = template
        if defaults is None or not required.issubset(values):
            return cls(ctx, **values)
        m = cls.__new__(cls)
        state = defaults.copy()  # field order, as validation leaves it
        state.update(values)
        object.__setattr__(m, "__dict__", state)
        # Every field set (as a new entry's usually are): the class's one set
        # of them, shared. Nothing removes a field from a fields set, and
        # adding one that is in it changes nothing.
        _set_fields_set(m, every if values.keys() == every else set(values))
        _set_extra(m, None)
        _set_private(m, None)
        Synchronized.lazily(m)  # entries are rarely locked: no lock until one is
        _set_ctx(m, ctx)
        _set_changed(m, {})
        return m

    @classmethod
    def model_fresh_rows(
        cls: Type[TReactiveModel], ctx: ContextRef, rows: List[Tuple[Any, ...]]
    ) -> List[TReactiveModel]:
        """``model_fresh(ctx, dict(zip(fields, row)))`` for each row, a value
        for every field in field order: the template is looked up once, and
        each row becomes its instance's fields as it is, with the class's one
        fields set (see ``model_fresh``)."""
        template = _FRESH.get(cls)
        if template is None or template[0] is not cls.__pydantic_fields__:
            template = _fresh_template(cls)
        _, defaults, _, every = template
        if defaults is None:
            return [cls(ctx, **dict(zip(cls.__pydantic_fields__, row))) for row in rows]
        names = tuple(defaults)
        new = cls.__new__
        set_dict = object.__setattr__
        built = []
        for row in rows:
            m = new(cls)
            set_dict(m, "__dict__", dict(zip(names, row)))
            _set_fields_set(m, every)
            _set_extra(m, None)
            _set_private(m, None)
            Synchronized.lazily(m)
            _set_ctx(m, ctx)
            _set_changed(m, {})
            built.append(m)
        return built

    def provide_context(
        self, ctx: Union["ReactiveModel", "weakref.ref[ReactiveContext]"]
    ) -> None:
        """Give tree a reference"""
        self._bind_context(ctx)
        # Once per bind, for the whole subtree: a model bound with changes it
        # already carries (one moved back into the tree) must not go unseen.
        live = self.ctx()
        if live is not None:
            live.state_attached(self)

    def _bind_context(
        self, ctx: Union["ReactiveModel", "weakref.ref[ReactiveContext]"]
    ) -> None:
        if isinstance(ctx, ReactiveModel):
            object.__setattr__(self, "ctx", ctx.ctx)
        else:
            object.__setattr__(self, "ctx", ctx)

        for field in self.__class__.model_fields:
            value = getattr(self, field)

            if isinstance(value, ReactiveModel):
                value._bind_context(self)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    if isinstance(item, ReactiveModel):
                        item._bind_context(self)
            elif isinstance(value, dict):
                for item in value.values():
                    if isinstance(item, ReactiveModel):
                        item._bind_context(self)

    def model_reset_changed(self, *keys: str, v: Optional[int] = None) -> None:
        """
        Reset the changed state, this will clear model_self_changed_fields.
        """

        if not keys and v is None:
            object.__setattr__(self, "model_self_changed_fields", {})
            return

        if not keys and v is not None:
            object.__setattr__(
                self,
                "model_self_changed_fields",
                {k: v2 for k, v2 in self.model_self_changed_fields.items() if v2 > v},
            )
            return

        for key in keys:
            if v is not None and self.model_self_changed_fields.get(key, -1) > v:
                continue

            self.model_self_changed_fields.pop(key, None)

    @property
    def model_changed_fields(self) -> Set[str]:
        """
        Return a dictionary of all changed fields.
        """

        changed_fields = set(self.model_self_changed_fields.keys())

        tracked = _tracked_fields(self.__class__)
        for field_name in self.__class__.__pydantic_fields__:
            field_value = self.__dict__[field_name]

            if isinstance(field_value, ReactiveModel) and field_value.model_has_changed:
                changed_fields.add(field_name)

            if field_value and field_name in tracked:
                if isinstance(field_value, (list, tuple)):
                    field_value_list = list(enumerate(field_value))
                elif isinstance(field_value, dict):
                    field_value_list = list(field_value.items())
                else:
                    continue

                for inner_field_index, inner_field_value in field_value_list:
                    if (
                        isinstance(inner_field_value, ReactiveModel)
                        and inner_field_value.model_has_changed
                    ):
                        changed_fields.add(field_name)
                        break

        return changed_fields

    @property
    def model_recursive_changeset(self) -> Dict[str, int]:
        changed_fields = self.model_self_changed_fields.copy()
        # Inlined "keep the newest id per key": a nested helper here was a
        # closure built on every call, and a call per key.
        get = changed_fields.get

        values = self.__dict__
        for field_name in _tracked_fields(self.__class__):
            field_value = values[field_name]

            if not field_value:
                continue

            # Handle direct nested state models.
            if isinstance(field_value, ReactiveModel) and (
                field_value_changes := field_value.model_recursive_changeset
            ):
                for key, value in field_value_changes.items():
                    key = f"{field_name}.{key}"
                    old = get(key, -1)
                    changed_fields[key] = value if value > old else old

                value = max(field_value_changes.values())
                old = get(field_name, -1)
                changed_fields[field_name] = value if value > old else old
                continue

            # Handle collection of nested state models.
            if isinstance(field_value, list):
                # Generate a matchable key for list changes
                field_values = zip(repeat("*"), field_value)
            elif isinstance(field_value, tuple):
                field_values = enumerate(field_value)  # Tuples have fixed indices
            elif isinstance(field_value, dict):
                field_values = field_value.items()
            else:
                continue

            for inner_field_index, inner_field_value in field_values:
                if isinstance(inner_field_value, ReactiveModel) and (
                    inner_field_value_changes
                    := inner_field_value.model_recursive_changeset
                ):
                    prefix = f"{field_name}.{inner_field_index}"
                    for key, value in inner_field_value_changes.items():
                        key = f"{prefix}.{key}"
                        old = get(key, -1)
                        changed_fields[key] = value if value > old else old

                    x = max(inner_field_value_changes.values())
                    for key in (prefix, field_name):
                        old = get(key, -1)
                        changed_fields[key] = x if x > old else old

        return changed_fields

    def model_latest_change(self) -> Optional[int]:
        """The newest change id in this model's subtree, ``None`` if none.

        Equal to ``max(model_recursive_changeset.values())`` over the same walk,
        without building the changeset: the client only needs the newest id of
        a branch, and the keys (an f-string per change, per level) were most of
        the walk's cost.
        """
        own = self.model_self_changed_fields
        latest = max(own.values()) if own else None

        values = self.__dict__
        for field_name in _tracked_fields(self.__class__):
            field_value = values[field_name]

            if not field_value:
                continue

            if isinstance(field_value, ReactiveModel):
                children: Iterable[Any] = (field_value,)
            elif isinstance(field_value, (list, tuple)):
                children = field_value
            elif isinstance(field_value, dict):
                children = field_value.values()
            else:
                continue

            for child in children:
                if isinstance(child, ReactiveModel):
                    child_latest = child.model_latest_change()
                    if child_latest is not None and (
                        latest is None or child_latest > latest
                    ):
                        latest = child_latest

        return latest

    @property
    def model_has_changed(self) -> bool:
        """Return True, when some field was changed."""

        if self.model_self_changed_fields:
            return True

        return bool(self.model_changed_fields)

    def model_has_changes(self, *fields: str) -> bool:
        """
        Check if the given fields have changed.
        If no fields are given, check if any field has changed.
        """

        if not fields:
            return self.model_has_changed

        # Ensure all fields exists
        for name in fields:
            if name not in self.__class__.__pydantic_fields__:
                raise AttributeError(f"Field {name} not available in this model")

        return any(name in self.model_self_changed_fields for name in fields)

    def model_set_changed(self, *fields: str) -> None:
        """Set fields as changed."""

        # Ensure all fields exists
        for name in fields:
            if name not in self.__class__.__pydantic_fields__:
                raise AttributeError(f"Field {name} not available in this model")

        ctx = self.ctx()

        if not ctx:
            return

        # Mark fields as changed
        for name in fields:
            self.model_self_changed_fields[name] = ctx.stamp_change(self, name)

        ctx.signal()

    def _model_value_is_comparable_type(self, value: Any) -> bool:
        # The most common value first (None is neither container below).
        if value is None:
            return True
        if isinstance(value, (list, set, tuple)):
            return all(self._model_value_is_comparable_type(i) for i in value)
        elif isinstance(value, dict):
            return all(
                (
                    self._model_value_is_comparable_type(k)
                    and self._model_value_is_comparable_type(v)
                )
                for k, v in value.items()
            )

        return isinstance(value, self.COMPARABLE_TYPES)

    @staticmethod
    def _model_value_is_actually_unchanged(value1: Any, value2: Any) -> bool:
        return value1 == value2

    @no_type_check
    def __setattr__(self, name, value) -> None:  # noqa: ANN001
        # One cached lookup per write: tracked fields -> Exclusive. Missing:
        # not a field (private attributes too) or an Untracked one.
        exclusive = _write_policy(self.__class__).get(name)
        if exclusive is None:
            super().__setattr__(name, value)
            return

        # Get original value
        original_value = self.__dict__[name]

        # Store changed value using pydantic
        super().__setattr__(name, value)

        # Don't use value parameter directly, as pydantic validation might have changed it
        # (when validate_assignment == True)
        current_value = self.__dict__[name]

        if (
            not exclusive
            and self._model_value_is_comparable_type(original_value)
            and self._model_value_is_comparable_type(current_value)
            and self._model_value_is_actually_unchanged(original_value, current_value)
        ):
            return

        # Store changed state
        if ctx := self.ctx():
            self.model_self_changed_fields[name] = ctx.stamp_change(self, name)
            ctx.signal()

    def model_write(self, name: str, value: Any) -> bool:
        """Set field ``name`` to ``value`` as an assignment would, cheaper:
        for a writer that projects already-typed values (an integration's
        state projection). ``True`` if it was a change.

        Exactly an assignment's outcome: an untracked field, an exclusive one,
        one an own ``__setattr__`` acts on, or a validated one given a value
        its validation would change go through ``setattr``; for the rest the
        value is stored the way pydantic stores it (instance dict and
        fields-set) and marked changed only if it differs. A validated field
        of a plain scalar type keeps a value of exactly that type, and turns
        an int into a float for a float field (see ``_typed_writes``).
        """
        cls = self.__class__
        plain = _PLAIN_WRITES.get(cls)
        if plain is None or plain[0] is not cls.__pydantic_fields__:
            plain = _plain_writes(cls)
        if name not in plain[1]:
            typed = plain[3].get(name)
            if typed is None or type(value) not in typed:
                before = self.model_self_changed_fields.get(name)
                setattr(self, name, value)
                return self.model_self_changed_fields.get(name) != before
            if type(value) is int and float in typed:
                if not -_EXACT_INT <= value <= _EXACT_INT:
                    before = self.model_self_changed_fields.get(name)
                    setattr(self, name, value)
                    return self.model_self_changed_fields.get(name) != before
                value = float(value)  # what validation stores
        d = self.__dict__
        original = d[name]
        d[name] = value
        self.__pydantic_fields_set__.add(name)  # as pydantic does, changed or not
        if original == value:
            # The comparable-type check, without its call for the values a
            # projection writes (numbers, strings, None).
            scalars = plain[2]
            if (
                type(original) in scalars
                or self._model_value_is_comparable_type(original)
            ) and (type(value) in scalars or self._model_value_is_comparable_type(value)):
                return False
        if ctx := self.ctx():
            self.model_self_changed_fields[name] = ctx.stamp_change(self, name)
            ctx.signal()
        return True

    def model_write_many(self, items: Iterable[Tuple[str, Any]]) -> int:
        """``model_write`` for each ``(name, value)``, in order: the same
        outcome per field, with the model's write policy and context read
        once and one signal for all of them. For a writer that describes a
        model at once (a printer's host and client facts, when it is built).
        Returns how many fields changed."""
        cls = self.__class__
        plain = _PLAIN_WRITES.get(cls)
        if plain is None or plain[0] is not cls.__pydantic_fields__:
            plain = _plain_writes(cls)
        _, direct, scalars, _typed = plain
        d = self.__dict__
        fields_set = self.__pydantic_fields_set__
        comparable = self._model_value_is_comparable_type
        ctx = self.ctx()
        changed = 0
        for name, value in items:
            if name not in direct:
                changed += self.model_write(name, value)
                # A validated assignment replaces the instance dict.
                d = self.__dict__
                fields_set = self.__pydantic_fields_set__
                continue
            original = d[name]
            d[name] = value
            fields_set.add(name)
            if (
                original == value
                and (type(original) in scalars or comparable(original))
                and (type(value) in scalars or comparable(value))
            ):
                continue
            changed += 1
            if ctx:
                self.model_self_changed_fields[name] = ctx.stamp_change(self, name)
        if changed and ctx:
            ctx.signal()
        return changed

    @classmethod
    def model_writer(cls, *names: str) -> Callable[..., int]:
        """``write(model, *values)``: ``model_write_many(zip(names, values))``
        on a ``cls`` model, as one generated function per class and names.
        Each field's write policy is settled when it is generated, so a call
        is straight-line code: no per-field lookup, no pairs built, the
        context read once, one signal. For a projection that writes the same
        fields of a model on every report. Returns how many fields changed."""
        key = (cls, names)
        entry = _WRITERS.get(key)
        if entry is None or entry[0] is not cls.__pydantic_fields__:
            entry = _WRITERS[key] = (cls.__pydantic_fields__, _writer(cls, names))
        return entry[1]

    def __getstate__(self) -> Dict[str, Any]:
        state = super().__getstate__()
        state["model_self_changed_fields"] = self.model_self_changed_fields.copy()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        super().__setstate__(state)

        if "model_self_changed_fields" in state:
            object.__setattr__(
                self, "model_self_changed_fields", state["model_self_changed_fields"]
            )
        else:
            object.__setattr__(self, "model_self_changed_fields", {})

    @classmethod
    def model_construct(
        cls: Type[TReactiveModel], *args: Any, **kwargs: Any
    ) -> TReactiveModel:
        """Construct an unvalidated instance"""

        m = cast(TReactiveModel, super().model_construct(*args, **kwargs))
        m.model_reset_changed()
        return m

    def model_post_init(self, __context: Any) -> None:
        super().model_post_init(__context)
        self.model_reset_changed()

    def model_update(
        self, other: TReactiveModel, exclude: Optional[Set[str]] = None
    ) -> None:
        for field_name, model_field in self.__class__.model_fields.items():
            if exclude and field_name in exclude:
                continue

            if field_name in other.__class__.model_fields:
                field_value = getattr(other, field_name)

                if isinstance(field_value, ReactiveModel):
                    field_value.provide_context(self)

                setattr(self, field_name, field_value)

    def __copy__(self: TReactiveModel) -> TReactiveModel:
        clone = cast(
            TReactiveModel,
            super().__copy__(),
        )
        object.__setattr__(
            clone, "model_self_changed_fields", self.model_self_changed_fields.copy()
        )
        return clone

    def __deepcopy__(
        self: TReactiveModel, memo: Optional[Dict[int, Any]] = None
    ) -> TReactiveModel:
        clone = cast(
            TReactiveModel,
            super().__deepcopy__(memo=memo),
        )
        object.__setattr__(
            clone, "model_self_changed_fields", self.model_self_changed_fields.copy()
        )
        return clone


#: ``class -> (fields, tracked field names)``: the fields the changeset walk
#: recurses into (``is_pydantic_change_detect_annotation``), in field order.
#: Computed once per class; the field dict is kept so a ``model_rebuild``
#: (which swaps it) invalidates the entry.
_TRACKED_FIELDS: Dict[type, Tuple[Any, Dict[str, None]]] = {}


def _tracked_fields(cls: Type["ReactiveModel"]) -> Dict[str, None]:
    fields = cls.__pydantic_fields__
    entry = _TRACKED_FIELDS.get(cls)
    if entry is not None and entry[0] is fields:
        return entry[1]
    tracked = dict.fromkeys(
        name
        for name, field_info in fields.items()
        if cls.is_pydantic_change_detect_annotation(field_info=field_info)
    )
    _TRACKED_FIELDS[cls] = (fields, tracked)
    return tracked


#: ``class -> (fields, {tracked field name: is Exclusive})``: what
#: ``__setattr__`` needs to know about a field, once per class instead of two
#: cached predicate calls and a private-attribute lookup on every write.
#: Untracked fields are absent. Keyed on the field dict like ``_TRACKED_FIELDS``.
_WRITE_POLICY: Dict[type, Tuple[Any, Dict[str, bool]]] = {}


def _write_policy(cls: Type["ReactiveModel"]) -> Dict[str, bool]:
    fields = cls.__pydantic_fields__
    entry = _WRITE_POLICY.get(cls)
    if entry is not None and entry[0] is fields:
        return entry[1]
    policy = {
        name: _is_exclusive(field_info)
        for name, field_info in fields.items()
        if not _is_untracked(field_info)
    }
    _WRITE_POLICY[cls] = (fields, policy)
    return policy


#: ``class -> (fields, names written directly, scalar types, typed
#: names)``: what ``model_write`` needs, in one lookup. Written directly:
#: tracked, not Exclusive, not acted on by an own ``__setattr__``, and the
#: model doesn't validate assignments. Typed names (a validating model's):
#: written directly given a value of one of their types (``_typed_writes``).
#: The scalar types are the comparable ones a value's type can be checked
#: against by identity (``COMPARABLE_TYPES`` decides). Keyed on the field
#: dict like ``_TRACKED_FIELDS``.
_PLAIN_WRITES: Dict[type, Tuple[Any, frozenset, frozenset, Dict[str, frozenset]]] = {}

#: The largest int a float holds exactly: within it, ``float(value)`` is what
#: a float field's validation stores; beyond, the assignment decides.
_EXACT_INT = 2**53


def intercepts(*names: str) -> Callable[[Callable], Callable]:
    """Declares that a model's own ``__setattr__`` acts on ``names`` only:
    ``model_write`` may write the class's other fields directly. A model
    whose ``__setattr__`` doesn't declare it is assigned field by field."""

    def mark(setter: Callable) -> Callable:
        setter.__intercepts__ = frozenset(names)  # type: ignore[attr-defined]
        return setter

    return mark


def _intercepted(cls: Type["ReactiveModel"]) -> Optional[frozenset]:
    """The names own ``__setattr__`` overrides act on, ``None`` for "any"."""
    names: Set[str] = set()
    for klass in cls.__mro__:
        if klass is ReactiveModel:
            break
        setter = klass.__dict__.get("__setattr__")
        if setter is None:
            continue
        acts_on = getattr(setter, "__intercepts__", None)
        if acts_on is None:
            return None
        names.update(acts_on)
    return frozenset(names)


def _plain_writes(
    cls: Type["ReactiveModel"],
) -> Tuple[Any, frozenset, frozenset, Dict[str, frozenset]]:
    fields = cls.__pydantic_fields__
    validates = cls.model_config.get("validate_assignment")
    intercepted = _intercepted(cls)
    writable = (
        ()
        if intercepted is None
        else [
            name
            for name, exclusive in _write_policy(cls).items()
            if not exclusive and name not in intercepted
        ]
    )
    direct = frozenset(() if validates else writable)
    typed = _typed_writes(cls, writable) if validates else {}
    scalars = frozenset(
        {type(None)}
        | {t for t in (int, float, str, bool) if issubclass(t, cls.COMPARABLE_TYPES)}
    )
    entry = _PLAIN_WRITES[cls] = (fields, direct, scalars, typed)
    return entry


_STR_CONFIG = (
    "str_strip_whitespace",
    "str_to_lower",
    "str_to_upper",
    "str_min_length",
    "str_max_length",
)


def _typed_writes(cls: Type["ReactiveModel"], names: Iterable[str]) -> Dict[str, frozenset]:
    """A validating model's fields whose validation is known for some value
    types: a field annotated ``X`` or ``Optional[X]``, ``X`` one of int,
    float, str and bool, with no constraints, validators or freezing. A
    value of exactly ``X`` (or ``None`` for an optional one) is stored as
    given, an int as ``float(value)`` for a float field. Any other value,
    or any field of a model with model validators, is assigned."""
    config = cls.model_config
    decorators = cls.__pydantic_decorators__
    if config.get("frozen") or any(
        (decorators.validators, decorators.root_validators, decorators.model_validators)
    ):
        return {}
    validated: Set[str] = set()
    for decorator in decorators.field_validators.values():
        validated.update(decorator.info.fields)
    if "*" in validated:
        return {}
    fields = cls.__pydantic_fields__
    typed: Dict[str, frozenset] = {}
    for name in names:
        info = fields[name]
        if name in validated or info.metadata or info.frozen:
            continue
        annotation = info.annotation
        kinds: Set[type] = set()
        if get_origin(annotation) is Union:
            args = get_args(annotation)
            if len(args) != 2 or type(None) not in args:
                continue
            kinds.add(type(None))
            annotation = args[0] if args[1] is type(None) else args[1]
        if annotation not in (int, float, str, bool):
            continue
        if annotation is str and any(config.get(key) for key in _STR_CONFIG):
            continue
        if annotation is float:
            if config.get("allow_inf_nan") is False:
                continue
            kinds.add(int)  # stored as float(value)
        kinds.add(annotation)
        typed[name] = frozenset(kinds)
    return typed


#: ``(class, names) -> (fields, writer)``: ``model_writer``'s functions, keyed
#: on the field dict like ``_TRACKED_FIELDS``.
_WRITERS: Dict[Tuple[type, Tuple[str, ...]], Tuple[Any, Callable[..., int]]] = {}


def _writer(cls: Type["ReactiveModel"], names: Tuple[str, ...]) -> Callable[..., int]:
    """The source of ``model_writer``'s function, compiled. Per field: the
    direct write (``model_write``'s, inline), behind a type test for a typed
    one, or a ``model_write`` call for one that is assigned."""
    fields = cls.__pydantic_fields__
    for name in names:
        if name not in fields:
            raise AttributeError(f"{cls.__name__} has no field {name!r}")
    _, direct, scalars, typed = _plain_writes(cls)
    params = [f"v{i}" for i in range(len(names))]
    lines = [
        f"def write(m, {', '.join(params)}):",
        "    d = m.__dict__",
        "    fs = m.__pydantic_fields_set__",
        "    ctx = m.ctx()",
        "    c = 0",
    ]
    namespace: Dict[str, Any] = {
        "S": scalars,
        "EXACT": _EXACT_INT,
    }

    def direct_write(name: str, v: str, indent: str) -> None:
        lines.extend(
            indent + line
            for line in (
                f"o = d[{name!r}]",
                f"d[{name!r}] = {v}",
                f"fs.add({name!r})",
                f"if not (o == {v} and (type(o) in S or m._model_value_is_comparable_type(o))"
                f" and (type({v}) in S or m._model_value_is_comparable_type({v}))):",
                "    c += 1",
                "    if ctx:",
                f"        m.model_self_changed_fields[{name!r}] = ctx.stamp_change(m, {name!r})",
            )
        )

    def assigned(name: str, v: str, indent: str) -> None:
        lines.extend(
            indent + line
            for line in (
                f"c += m.model_write({name!r}, {v})",
                # A validated assignment replaces the instance dict.
                "d = m.__dict__",
                "fs = m.__pydantic_fields_set__",
            )
        )

    for i, (name, v) in enumerate(zip(names, params)):
        if name in direct:
            direct_write(name, v, "    ")
            continue
        kinds = typed.get(name)
        if kinds is None:
            assigned(name, v, "    ")
            continue
        namespace[f"K{i}"] = kinds
        test = f"type({v}) in K{i}"
        if float in kinds:
            # An int is stored as float(value), within the exact range.
            lines.append(
                f"    if type({v}) is int and -EXACT <= {v} <= EXACT: {v} = float({v})"
            )
            test = f"type({v}) is not int and {test}"
        lines.append(f"    if {test}:")
        direct_write(name, v, "        ")
        lines.append("    else:")
        assigned(name, v, "        ")
    lines.extend(("    if c and ctx:", "        ctx.signal()", "    return c"))
    code = compile("\n".join(lines), f"<model_writer {cls.__qualname__}{names}>", "exec")
    exec(code, namespace)
    return namespace["write"]


#: ``class -> (fields, defaults by field in field order or None, required
#: names, the set of every field name)``: ``model_fresh``'s template. ``None``: the class can't be built
#: from it (see ``model_fresh``). Keyed on the field dict like
#: ``_TRACKED_FIELDS``.
_FRESH: Dict[type, Tuple[Any, Optional[Dict[str, Any]], frozenset, Set[str]]] = {}

_PLAIN_DEFAULTS = (type(None), int, float, str, bool, enum.Enum)


def _fresh_template(
    cls: Type["ReactiveModel"],
) -> Tuple[Any, Optional[Dict[str, Any]], frozenset, Set[str]]:
    fields = cls.__pydantic_fields__
    decorators = cls.__pydantic_decorators__
    plain = (
        cls.model_post_init is ReactiveModel.model_post_init
        and not cls.__private_attributes__
        and cls.model_config.get("extra") != "allow"
        and not any(
            (
                decorators.validators,
                decorators.field_validators,
                decorators.root_validators,
                decorators.model_validators,
            )
        )
    )
    defaults: Optional[Dict[str, Any]] = {}
    required = []
    for name, info in fields.items():
        if info.is_required():
            required.append(name)
            defaults[name] = None  # a placeholder: values always has it
        elif info.default_factory is None and isinstance(info.default, _PLAIN_DEFAULTS):
            defaults[name] = info.default
        else:
            plain = False
    entry = _FRESH[cls] = (
        fields,
        defaults if plain else None,
        frozenset(required),
        set(defaults),
    )
    return entry


# model_fresh's slot writes, straight through the slots' descriptors (what
# object.__setattr__ finds for them, without the lookup).
_set_fields_set = BaseModel.__dict__["__pydantic_fields_set__"].__set__
_set_extra = BaseModel.__dict__["__pydantic_extra__"].__set__
_set_private = BaseModel.__dict__["__pydantic_private__"].__set__
_set_ctx = ReactiveModel.__dict__["ctx"].__set__
_set_changed = ReactiveModel.__dict__["model_self_changed_fields"].__set__
