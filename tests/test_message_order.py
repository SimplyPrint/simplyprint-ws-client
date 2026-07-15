from simplyprint_ws_client import (
    Client,
    PrinterStatus,
)
from simplyprint_ws_client.core.protocol.messages import JobInfoMsg, StateChangeMsg


def test_message_order_simple(client: Client):
    msgs = client.consume()
    assert len(msgs) == 0

    client.printer.job_info.started = True
    client.printer.status = PrinterStatus.PRINTING

    # Assert the order of messages
    msgs = client.consume()
    assert len(msgs) == 2
    assert msgs[0].__class__ == JobInfoMsg
    assert msgs[1].__class__ == StateChangeMsg
