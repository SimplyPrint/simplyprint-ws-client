__all__ = [
    "configure",
    "autoconfigure",
    "autoconfigure_class_dict",
    "debug",
    "autowire",
    "AutowireClientMeta",
]

import asyncio
import functools
import inspect
from abc import ABCMeta
from functools import partial
from typing import (
    Type,
    get_origin,
    Hashable,
    Dict,
    Optional,
    get_args,
    Literal,
    TYPE_CHECKING,
    Callable,
)

from pydantic import BaseModel

from simplyprint_ws_client.core.protocol.messages import ServerMsgKind, DemandMsgKind
from simplyprint_ws_client.core.protocol.models import ServerMsgType, DemandMsgType
from simplyprint_ws_client.events.event_bus_listeners import (
    EventBusListenerOptions,
)

if TYPE_CHECKING:
    from simplyprint_ws_client.core.client import Client  # noqa: F401
    from simplyprint_ws_client.core.state import PrinterState  # noqa: F401

try:
    from typing import Unpack
except ImportError:
    from typing_extensions import Unpack


def configure(
    event: Optional[Hashable] = None, **kwargs: Unpack[EventBusListenerOptions]
):
    """Map a function as a callback for an event."""

    def decorator(func):
        if not hasattr(func, "_event_bus_event"):
            func._event_bus_event = None

        if not hasattr(func, "_event_bus_wrap"):
            func._event_bus_wrap = False

        if not hasattr(func, "_event_bus_listeners_args"):
            func._event_bus_listeners_args = {}

        func._event_bus_listeners_args.update(kwargs)
        func._event_bus_event = event or func._event_bus_event

        return func

    return decorator


def autoconfigure(attr: Callable):
    """Automatically configure a function based on its signature."""
    if isinstance(attr, partial) or isinstance(attr, functools.partialmethod):
        attr = attr.func

    if not inspect.isfunction(attr):
        raise ValueError("Expected a function.")

    signature = inspect.signature(attr)
    parameters = dict(signature.parameters.copy())
    parameters.pop("self", None)

    if len(parameters) == 0:
        attr._event_bus_wrap = True

    name = attr.__name__

    msg_lookup, msg_data_lookup, demand_lookup = build_autoconfiguration_state()

    # Initially, determine event type based on function name.
    if name.startswith("on_"):
        event_name: str = name[3:]

        if event_name.upper() in ServerMsgType.__members__:
            event_guess = ServerMsgType(event_name)
            attr._event_bus_event = event_guess
            return

        elif event_name.upper() in DemandMsgType.__members__:
            event_guess = DemandMsgType(event_name)
            attr._event_bus_event = event_guess
            return

    # We extract event information given one argument.
    if len(parameters) != 1:
        return

    param_type = next(iter(parameters.values())).annotation

    # Only consider first argument type.
    if not isinstance(param_type, type):
        return

    if msg_cls := msg_lookup.get(param_type):
        attr._event_bus_event = msg_cls
        return

    if demand_cls := demand_lookup.get(param_type):
        attr._event_bus_event = demand_cls
        return

    if data_cls := msg_data_lookup.get(param_type):
        attr._event_bus_event = data_cls
        return


@functools.lru_cache()
def build_autoconfiguration_state():
    demand_lookup: Dict[Type[BaseModel], DemandMsgType] = {}
    msg_lookup: Dict[Type[BaseModel], ServerMsgType] = {}
    msg_data_lookup: Dict[Type[BaseModel], ServerMsgType] = {}

    for m in set(get_args(ServerMsgKind)):
        type_annotation = m.model_fields["type"].annotation

        if get_origin(type_annotation) is not Literal:
            continue

        msg_type = get_args(type_annotation)[0]

        if not isinstance(msg_type, ServerMsgType):
            continue

        msg_lookup[m] = msg_type

        data_annotation = m.model_fields["data"].annotation

        if isinstance(data_annotation, type) and issubclass(data_annotation, BaseModel):
            msg_data_lookup[data_annotation] = msg_type

    for m in set(get_args(DemandMsgKind)):
        demand_annotation = m.model_fields["demand"].annotation

        if get_origin(demand_annotation) is not Literal:
            continue

        demand_type = get_args(demand_annotation)[0]

        if not isinstance(demand_type, DemandMsgType):
            continue

        demand_lookup[m] = demand_type

    return msg_lookup, msg_data_lookup, demand_lookup


def autoconfigure_class_dict(class_dict: dict):
    """
    # autofill opts based on:
    # function name
    # function arguments
    """
    for name in class_dict:
        attr = class_dict[name]

        # Only consider functions.
        if not inspect.isfunction(attr) and not isinstance(attr, partial):
            continue

        autoconfigure(attr)


def debug(cls: Type["Client"]):
    # Show all configured functions
    for name in dir(cls):
        attr = getattr(cls, name)

        if not inspect.isfunction(attr):
            continue

        if hasattr(attr, "_event_bus_event"):
            print(f"{name} => {getattr(attr, '_event_bus_event')}")


def autowire(client: "Client"):
    """Instrument and instance of a client based on its methods."""
    for name in dir(client.__class__):
        attr = getattr(client.__class__, name)

        if not hasattr(attr, "_event_bus_event"):
            continue

        e = getattr(attr, "_event_bus_event")
        kwargs = {}

        if hasattr(attr, "_event_bus_listeners_args"):
            kwargs = getattr(attr, "_event_bus_listeners_args")

        func = getattr(client, name)

        if (
            hasattr(attr, "_event_bus_wrap")
            and getattr(attr, "_event_bus_wrap") is True
        ):
            if asyncio.iscoroutinefunction(func):

                async def func(*args, inner_func=func, **_):
                    return await inner_func()
            else:

                def func(*args, inner_func=func, **_):
                    return inner_func()

        client.event_bus.on(e, func, **kwargs)


class AutowireClientMeta(ABCMeta):
    """On type creation, autoconfigure the class dictionary."""

    def __new__(cls, name, bases, class_dict, **kwargs):
        autoconfigure_class_dict(class_dict)
        return super().__new__(cls, name, bases, class_dict, **kwargs)
