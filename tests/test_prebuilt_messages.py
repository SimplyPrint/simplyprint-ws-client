"""``ClientMsg.prebuilt(data)`` is ``cls(data)`` without validation: the same
fields in the same order, fields set, private and extra attributes, frames and
equality, for every state message kind; a class that needs validation is built
the validated way."""

from typing import Literal

from pydantic import PrivateAttr, field_validator

from simplyprint_ws_client.core import client as client_module
from simplyprint_ws_client.core.protocol.messages import (
    _PREBUILT,
    ClientMsg,
    PeripheralMsg,
)
from simplyprint_ws_client.core.protocol.models import ClientMsgType, WriteOutcome


def _whole(m: ClientMsg) -> tuple:
    return (
        type(m),
        list(m.__dict__.items()),
        m.__pydantic_fields_set__,
        m.__pydantic_private__,
        m.__pydantic_extra__,
        m.model_dump(),
        m.model_dump(mode="json"),
        m.model_dump_json(),
    )


_DATA = {"fan:1": {"v": 0.5, "a": True}, "n": [1, None, "x"], "t": 21.5}


def test_every_state_message_is_its_validated_build():
    for kind, _ in client_module._MESSAGE_KINDS:
        built = kind.prebuilt(dict(_DATA))
        assert _PREBUILT[kind][1] is not None, kind  # not the fallback
        assert _whole(built) == _whole(kind(dict(_DATA))), kind
        assert built == kind(dict(_DATA))


def test_a_prebuilt_message_owns_its_fields_set_and_private():
    first, second = PeripheralMsg.prebuilt({"a": 1}), PeripheralMsg.prebuilt({"b": 2})
    first.for_client = "unique"  # the manager addresses a multi-printer frame
    first.record_write(WriteOutcome.WRITTEN)
    assert second.__pydantic_fields_set__ == {"type", "data"}
    assert second.write_outcome is None
    assert second.model_dump(mode="json") == PeripheralMsg({"b": 2}).model_dump(
        mode="json"
    )
    assert first.model_dump(mode="json")["for"] == "unique"


class _Validated(ClientMsg[Literal[ClientMsgType.PERIPHERAL]]):
    @field_validator("data")
    @classmethod
    def _doubled(cls, data):
        return {key: value * 2 for key, value in data.items()}


class _OwnPostInit(ClientMsg[Literal[ClientMsgType.PERIPHERAL]]):
    _seen: list = PrivateAttr(default_factory=list)

    def model_post_init(self, context) -> None:
        self._seen.append("post")


def test_a_class_that_needs_validation_is_validated():
    assert _Validated.prebuilt({"a": 1}).data == {"a": 2}
    built = _OwnPostInit.prebuilt({"a": 1})
    assert built._seen == ["post"]
    assert _PREBUILT[_Validated][1] is None and _PREBUILT[_OwnPostInit][1] is None
