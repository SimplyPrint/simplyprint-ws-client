from simplyprint_ws_client import Client


def commit_pending(client: Client) -> list:
    pending = client.pending_messages()
    for item in pending:
        client.commit_message(item)
    return [item.message for item in pending]
