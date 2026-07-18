"""
Events sent to and from a connection.
"""

from simplyprint_ws_client.events import Event


class SimplyPrintConnectionEvent(Event): ...


class SimplyPrintConnectionIncomingEvent(SimplyPrintConnectionEvent): ...


class SimplyPrintConnectionOutgoingEvent(SimplyPrintConnectionEvent): ...


class SimplyPrintConnectionEstablishedEvent(SimplyPrintConnectionEvent):
    v: int

    def __init__(self, v: int):
        self.v = v


class SimplyPrintConnectionLostEvent(SimplyPrintConnectionEvent):
    v: int

    def __init__(self, v: int):
        self.v = v


class SimplyPrintConnectionSuspectEvent(SimplyPrintConnectionEvent): ...
