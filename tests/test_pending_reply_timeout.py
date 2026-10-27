"""A multi-printer add or remove the server never answers doesn't strand the printer."""

from datetime import datetime

import pytest

from simplyprint_ws_client import Client, ClientContext, PrinterConfig
from simplyprint_ws_client.core.client import PENDING_REPLY_TIMEOUT, ClientState
from simplyprint_ws_client.core.protocol.connection import ConnectionMode


def _client(state: ClientState, sent_ago, monkeypatch) -> Client:
    client = Client(PrinterConfig.get_new(), context=ClientContext())
    sent = []

    async def send(msg, skip_dispatch=False):
        sent.append(type(msg).__name__)

    monkeypatch.setattr(client, "send", send)
    client.sent = sent
    client.state = state
    client._pending_action_ts = datetime.now() - sent_ago
    return client


@pytest.mark.asyncio
async def test_an_unanswered_remove_lets_the_printer_be_added_again(monkeypatch):
    client = _client(ClientState.PENDING_REMOVED, PENDING_REPLY_TIMEOUT, monkeypatch)
    # The device link is back: the scheduler wants the printer added.
    await client.ensure_added(ConnectionMode.MULTI)
    assert client.state == ClientState.PENDING_ADDED
    assert client.sent == ["MultiPrinterAddConnectionMsg"]


@pytest.mark.asyncio
async def test_an_unanswered_add_is_sent_again(monkeypatch):
    client = _client(ClientState.PENDING_ADDED, PENDING_REPLY_TIMEOUT, monkeypatch)
    await client.ensure_added(ConnectionMode.MULTI)
    assert client.sent == ["MultiPrinterAddConnectionMsg"]


@pytest.mark.asyncio
async def test_a_reply_still_due_is_waited_for(monkeypatch):
    client = _client(ClientState.PENDING_REMOVED, PENDING_REPLY_TIMEOUT / 2, monkeypatch)
    await client.ensure_added(ConnectionMode.MULTI)
    assert client.state == ClientState.PENDING_REMOVED
    assert client.sent == []
