__all__ = ["Config", "PrinterConfig"]

import json
import uuid
from abc import ABC, abstractmethod
from typing import ClassVar, Optional, Tuple

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

    @classmethod
    def update_dict_keys(cls, data: dict):
        """Modify incoming data to match the keys of the config."""
        pk_key, sk_key = cls.keys()

        if "pk" in data:
            data[pk_key] = data.pop("pk")

        if "sk" in data:
            data[sk_key] = data.pop("sk")

        return data

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
    def pk(self) -> int:
        """Primary key for the config."""
        return int(getattr(self, self.keys()[0]))

    @property
    def sk(self) -> str:
        """Secondary key for the config."""
        return str(getattr(self, self.keys()[1]))

    @property
    def key(self) -> TKey:
        return self.pk, self.sk

    @staticmethod
    @abstractmethod
    def keys() -> tuple:
        """Return the keys of the config."""
        raise NotImplementedError()

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
    #: MAC -- see ``stable_hardware_id``.
    mac: Optional[str] = None

    @staticmethod
    def keys() -> tuple:
        return "id", "token"

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
    #   * ``stable_hardware_id`` is the *device* identity (serial/guid/MAC) used
    #     only to correlate a re-discovered physical printer back to its slot.
    # Keeping them apart means swapping the printer in a slot keeps the slot's id,
    # while re-discovery still finds the right slot by hardware identity.

    #: Config fields that may carry the device's network address, in match
    #: priority order. The identity/reconcile seams probe these to resolve a MAC
    #: fallback and to de-duplicate by address when neither side has a hardware
    #: id. A brand whose config reaches its device through a differently-named
    #: field extends this tuple on its config class.
    network_address_fields: ClassVar[Tuple[str, ...]] = ("host", "local_ip")

    def stable_hardware_id(self) -> Optional[str]:
        """The brand's own immutable device id for this printer, or ``None``.

        Override per brand to return the device's hardware id (serial, board
        uniqueId, system guid, ...). When a brand has none, the neutral
        :attr:`mac` -- captured at onboarding -- is the fallback the matching seam
        applies, so a re-discovered device is still found without a serial. This
        is *hardware* identity for correlation -- never a credential (access codes
        / auth keys are not identity) and never the ``unique_id`` slot ref.
        """
        return None
