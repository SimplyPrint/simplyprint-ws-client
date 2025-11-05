"""
Core Protocols and Abstractions

Base classes for building printer integrations.
"""

__all__ = ["Protocol", "Client", "Printer"]

from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from dataclasses import dataclass

TConfig = TypeVar("TConfig")


@dataclass
class Config:
    """Base configuration class"""
    pass


class Protocol(ABC):
    """
    Base class for communication protocols.

    Protocols handle external I/O (HTTP, WebSocket, MQTT, etc.)
    and emit events to the application layer.

    Usage:
        class MQTTProtocol(Protocol):
            async def start(self):
                await self.connect()

            async def stop(self):
                await self.disconnect()
    """

    @abstractmethod
    async def start(self):
        """Start the protocol (connect, subscribe, etc.)"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the protocol gracefully"""
        pass


class Client(ABC):
    """
    Base class for printer clients.

    Clients handle business logic and coordinate protocols.

    Usage:
        class MyClient(Client):
            protocol: MyProtocol = Injected

            async def start(self):
                await self.protocol.start()

            async def stop(self):
                await self.protocol.stop()
    """

    @abstractmethod
    async def start(self):
        """Start the client"""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the client"""
        pass


class Printer(Client, Generic[TConfig]):
    """
    Base class for printer implementations.

    Combines configuration, protocols, and business logic.

    Usage:
        @dataclass
        class MyConfig(Config):
            host: str
            port: int

        class MyPrinter(Printer[MyConfig]):
            def __init__(self, config: MyConfig):
                self.config = config

            async def start(self):
                # Start protocols
                pass

            async def stop(self):
                # Stop protocols
                pass
    """

    config: TConfig

    @abstractmethod
    async def start(self):
        pass

    @abstractmethod
    async def stop(self):
        pass
