"""
Events sent to and from a connection.
"""

from typing import Optional

from simplyprint_ws_client.events import Event


class SimplyPrintConnectionEvent(Event): ...


class SimplyPrintConnectionIncomingEvent(SimplyPrintConnectionEvent): ...


class SimplyPrintConnectionOutgoingEvent(SimplyPrintConnectionEvent): ...


class SimplyPrintConnectionEstablishedEvent(SimplyPrintConnectionEvent):
    v: int
    #: Printers on the hub that just opened (MULTI only): the client spreads
    #: its add over a window this size scales. ``None`` for a SINGLE link.
    hub_size: Optional[int]

    def __init__(self, v: int, hub_size: Optional[int] = None):
        self.v = v
        self.hub_size = hub_size


class SimplyPrintConnectionLostEvent(SimplyPrintConnectionEvent):
    v: int

    def __init__(self, v: int):
        self.v = v


class SimplyPrintConnectionSuspectEvent(SimplyPrintConnectionEvent): ...
