import base64
import contextlib
import json
from typing import Optional, Union, final

import aiohttp
from aiohttp import ClientTimeout
from yarl import URL

from simplyprint_ws_client.core.api.url_builder import SimplyPrintEndpoints
from simplyprint_ws_client.const import VERSION


class SimplyPrintApiError(Exception):
    """A SimplyPrint HTTP API call failed (non-200 response)."""


def _company_id_from_token(action_token: str) -> int:
    """Extract the company id from an action token (a JWT) without verifying it.

    Hacky but deliberate: the token is opaque to the client except for this one
    routing value the API path needs.
    """
    payload = json.loads(
        base64.b64decode(action_token.split(".")[1] + "===").decode("utf-8")
    )
    return payload["company"]


@final
class SimplyPrintApi:
    """HTTP operations scoped to one app's immutable endpoints."""

    def __init__(self, endpoints: SimplyPrintEndpoints) -> None:
        self.endpoints = endpoints
        self.api_url = endpoints.api_url

    async def post_snapshot(
        self,
        snapshot_id: str,
        image_data: bytes,
        endpoint: Union[str, URL, None] = None,
    ):
        if endpoint is None:
            endpoint = self.api_url / "jobs" / "ReceiveSnapshot"

        data = {
            "id": snapshot_id,
            "image": base64.b64encode(image_data).decode("utf-8"),
        }

        headers = {"User-Agent": f"simplyprint-ws-client/{VERSION}"}

        async with aiohttp.ClientSession() as session:
            async with session.post(
                str(endpoint),
                data=data,
                headers=headers,
                timeout=ClientTimeout(total=45),
            ) as response:
                if response.status != 200:
                    raise SimplyPrintApiError(
                        f"Failed to post snapshot: {await response.text()}"
                    )

    async def post_logs(
        self,
        printer_id: int,
        token: str,
        main_log_file: Optional[str] = None,
        plugin_log_file: Optional[str] = None,
        serial_log_file: Optional[str] = None,
    ):
        # Request /printers/ReceiveLogs with the token as post data
        # And each of the files as multipart/form-data

        endpoint = self.api_url / "printers" / "ReceiveLogs" % {"pid": printer_id}

        data = {
            "token": token,
        }

        headers = {"User-Agent": f"simplyprint-ws-client/{VERSION}"}

        # The handles must stay open until aiohttp has streamed the upload, and
        # must always be closed afterwards.
        with contextlib.ExitStack() as stack:
            if main_log_file:
                data["main"] = stack.enter_context(open(main_log_file, "r"))

            if plugin_log_file:
                data["plugin_log"] = stack.enter_context(open(plugin_log_file, "r"))

            if serial_log_file:
                data["serial_log"] = stack.enter_context(open(serial_log_file, "r"))

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    str(endpoint),
                    data=data,
                    headers=headers,
                    timeout=ClientTimeout(total=45),
                ) as response:
                    if response.status != 200:
                        raise SimplyPrintApiError(
                            f"Failed to post logs: {await response.text()}"
                        )

                    return await response.json()

    async def clear_bed(
        self,
        printer_id: int,
        action_token: str,
        success: bool,
        rating: Optional[int] = None,
    ):
        headers = {
            "X-Action-Token": action_token,
        }

        endpoint = (
            self.api_url
            / str(_company_id_from_token(action_token))
            / "printers"
            / "actions"
            / "ClearBed"
            % {"pid": printer_id}
        )

        data = {
            "success": success,
            "rating": rating,
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                str(endpoint),
                json=data,
                headers=headers,
                timeout=ClientTimeout(total=45),
            ) as response:
                if response.status != 200:
                    raise SimplyPrintApiError(
                        f"Failed to clear bed: {await response.text()}"
                    )

                return await response.json()

    async def start_next_print(self, printer_id: int, action_token: str):
        headers = {
            "X-Action-Token": action_token,
        }

        data = {
            "pid": printer_id,
            "next_queue_item": True,
        }

        endpoint = (
            self.api_url
            / str(_company_id_from_token(action_token))
            / "printers"
            / "actions"
            / "CreateJob"
        )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                str(endpoint),
                json=data,
                headers=headers,
                timeout=ClientTimeout(total=45),
            ) as response:
                if response.status != 200:
                    raise SimplyPrintApiError(
                        f"Failed to start next print: {await response.text()}"
                    )

                return await response.json()
