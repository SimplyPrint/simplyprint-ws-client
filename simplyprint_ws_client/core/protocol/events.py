"""
Events sent to and from a connection.
"""

from simplyprint_ws_client.events import Event


class CloudConnectionEvent(Event): ...


class CloudConnectionIncomingEvent(CloudConnectionEvent): ...


class CloudConnectionOutgoingEvent(CloudConnectionEvent): ...


class CloudConnectionEstablishedEvent(CloudConnectionEvent):
    v: int

    def __init__(self, v: int):
        self.v = v


class CloudConnectionLostEvent(CloudConnectionEvent):
    v: int

    def __init__(self, v: int):
        self.v = v


class CloudConnectionSuspectEvent(CloudConnectionEvent): ...
