__all__ = ["Config", "PrinterConfig"]

import json
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Tuple

from pydantic import BaseModel

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

TKey = Tuple[int, str]


# TODO: Replace with a proper ORM-component
# And centralize the config management file with tables, versioning, singleton instances (settings)
# and more.
class Config(ABC):
    """Config Entity interface for persistence."""

    def __repr__(self):
        return f"{self.__class__.__name__}({self.as_dict()})"

    def __eq__(self, other: object) -> bool:
        """Each instance is unique."""
        if isinstance(other, self.__class__):
            return id(self) == id(other)

        return False

    def __hash__(self) -> int:
        return hash(id(self))

    @classmethod
    def make_hashable(cls):
        """Override standard hash and eq methods from dataclass + pydantic etc."""
        cls.__hash__ = Config.__hash__
        cls.__eq__ = Config.__eq__

    def partial_eq(self, config: Optional["Config"] = None, **kwargs) -> bool:
        """Check if the other config is partially equal to this one."""
        data = self.as_dict()

        if config is not None:
            kwargs.update(config.as_dict())

        for key, value in kwargs.items():
            if key not in data or data[key] != value:
                return False

        return True

    @property
    @abstractmethod
    def pk(self) -> int:
        """Primary key for the config."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def sk(self) -> str:
        """Secondary key for the config."""
        raise NotImplementedError()

    @property
    def key(self) -> TKey:
        return self.pk, self.sk

    @abstractmethod
    def is_empty(self) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def as_dict(self) -> dict:
        raise NotImplementedError()

    def as_json(self) -> str:
        return json.dumps(self.as_dict())

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Self:
        raise NotImplementedError()


class PrinterConfig(BaseModel, Config):
    """Configuration object for printers (a pydantic model).

    Every concrete config is a pydantic model: brands subclass this directly and
    add their own fields. (``id``/``token`` stay nullable: a not-yet-keyed config
    legitimately carries ``None`` for them.)
    """

    id: Optional[int]
    token: Optional[str]

    name: Optional[str] = None
    in_setup: Optional[bool] = None
    short_id: Optional[str] = None
    public_ip: Optional[str] = None
    unique_id: Optional[str] = None
    #: The device's MAC address, captured at onboarding when the brand exposes no
    #: serial/guid. A neutral, stable hardware identifier (separate from the slot's
    #: ``unique_id``) so a re-discovered printer can be matched to its config by
    #: MAC -- see ``hardware_identity``.
    mac: Optional[str] = None
    #: A user-supplied webcam URL that overrides whatever camera the device
    #: itself advertises. Every printer supports it: the base client resolves
    #: it ahead of the brand's own camera probe (see
    #: ``PrinterClient.update_camera_uri``), and the shared presentation kit
    #: exposes it as an editable field.
    custom_webcam_url: Optional[str] = None

    @property
    def pk(self) -> int:
        return int(self.id)

    @property
    def sk(self) -> str:
        return str(self.token)

    def is_empty(self) -> bool:
        data = {k: v for k, v in self.as_dict().items() if v is not None}
        return self.is_default() and len(set(data.keys()) - {"id", "token"}) == 0

    def as_dict(self) -> dict:
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        return cls.model_validate(data)

    def is_pending(self) -> bool:
        return self.id == 0 or self.id is None or self.in_setup

    def is_default(self) -> bool:
        return self.is_pending() and (self.token is None or len(self.token) < 2)

    @classmethod
    def get_blank(cls) -> Self:
        return cls(id=0, token="0")

    @classmethod
    def get_new(cls) -> Self:
        return cls(id=0, token="0", unique_id=str(uuid.uuid4()))

    # Two separate identities, on purpose:
    #   * ``unique_id`` is the *slot* reference -- a stable UUID assigned once and
    #     never re-keyed. It is the ``client_list`` key and the backend's handle,
    #     and it survives both an IP change and a physical-printer swap.
    #   * ``hardware_identity`` is the *device* identity (serial/guid/MAC) used
    #     only to correlate a re-discovered physical printer back to its slot.
    # Keeping them apart means swapping the printer in a slot keeps the slot's id,
    # while re-discovery still finds the right slot by hardware identity.

    def hardware_identity(self) -> Optional[str]:
        """The device identity used to correlate discovery with this config.

        The base-owned MAC is the neutral fallback. Brands with a stronger
        immutable serial/guid override this method and fall back to ``super()``.
        The slot's :attr:`unique_id` is deliberately unrelated, and credentials
        such as access codes/auth keys are never identity.
        """
        return self.mac

    def network_addresses(self) -> Tuple[str, ...]:
        """Network addresses that can reach this printer, in preference order.

        The base config owns no device endpoint. A concrete integration returns
        its actual address values directly instead of publishing attribute names
        for another subsystem to inspect.
        """
        return ()

    def primary_network_address(self) -> Optional[str]:
        """The preferred non-empty device address, if the integration has one."""
        return next((address for address in self.network_addresses() if address), None)

    def set_webcam_url(self, value: str | int | float | bool | None) -> None:
        """Set or clear the user-supplied camera override."""

        self.custom_webcam_url = None if value is None or value == "" else str(value)
