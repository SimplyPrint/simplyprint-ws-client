"""A tiny registry for *app-level* (non-per-printer) inbound server messages.

Most inbound messages are routed to a specific printer client by ``for_client``
(see :class:`~simplyprint_ws_client.core.manager.ClientView`). A few concern the
whole process rather than any one printer -- e.g. a relayed integration webhook
delivered to the one session that registered for it. Those have no ``for_client``
and would otherwise be dropped by the per-printer router.

This module lets app code (outside ``core``) register a single async handler per
:class:`ServerMsgType`. ``ClientView.emit`` consults the registry and, for a
matching top-level message, invokes the handler exactly once instead of
per-printer routing. The registry is deliberately generic (no brand/integration
names leak into ``core``); the app decides what each type means.
"""

from __future__ import annotations

from typing import Awaitable, Callable, Dict, Optional

from simplyprint_ws_client.core.protocol.models import ServerMsgType

__all__ = [
    "register_app_message_handler",
    "unregister_app_message_handler",
    "get_app_message_handler",
]

#: ``ServerMsgType`` -> ``async handler(msg, v)``. One handler per type: these
#: messages address the process, so a second registration is a bug, not a fan-out.
_handlers: Dict[ServerMsgType, Callable[..., Awaitable[None]]] = {}


def register_app_message_handler(
    msg_type: ServerMsgType, handler: Callable[..., Awaitable[None]]
) -> None:
    """Route ``msg_type`` to ``handler`` instead of per-printer delivery."""
    _handlers[msg_type] = handler


def unregister_app_message_handler(msg_type: ServerMsgType) -> None:
    """Forget any handler for ``msg_type`` (no-op when none is registered)."""
    _handlers.pop(msg_type, None)


def get_app_message_handler(
    msg_type: ServerMsgType,
) -> Optional[Callable[..., Awaitable[None]]]:
    """The handler for ``msg_type``, or ``None`` for normal per-printer routing."""
    return _handlers.get(msg_type)
