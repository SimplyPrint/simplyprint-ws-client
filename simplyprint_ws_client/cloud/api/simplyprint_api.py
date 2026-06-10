import base64
import json
from typing import Optional, Union

import aiohttp
from aiohttp import ClientTimeout
from yarl import URL

from simplyprint_ws_client.cloud.api.url_builder import SimplyPrintURL
from simplyprint_ws_client.const import VERSION


class SimplyPrintApi:
    @staticmethod
    async def post_snapshot(
        snapshot_id: str, image_data: bytes, endpoint: Union[str, URL, None] = None
    ):
        if endpoint is None:
            endpoint = SimplyPrintURL().api_url / "jobs" / "ReceiveSnapshot"

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
                    raise Exception(f"Failed to post snapshot: {await response.text()}")

    @staticmethod
    async def post_logs(
        printer_id: int,
        token: str,
        main_log_file: Optional[str] = None,
        plugin_log_file: Optional[str] = None,
        serial_log_file: Optional[str] = None,
    ):
        # Request /printers/ReceiveLogs with the token as post data
        # And each of the files as multipart/form-data

        endpoint = (
            SimplyPrintURL().api_url / "printers" / "ReceiveLogs" % {"pid": printer_id}
        )

        data = {
            "token": token,
        }

        headers = {"User-Agent": f"simplyprint-ws-client/{VERSION}"}

        if main_log_file:
            data["main"] = open(main_log_file, "r")

        if plugin_log_file:
            data["plugin_log"] = open(plugin_log_file, "r")

        if serial_log_file:
            data["serial_log"] = open(serial_log_file, "r")

        async with aiohttp.ClientSession() as session:
            async with session.post(
                str(endpoint),
                data=data,
                headers=headers,
                timeout=ClientTimeout(total=45),
            ) as response:
                if response.status != 200:
                    raise Exception(f"Failed to post logs: {await response.text()}")

                return await response.json()

    @staticmethod
    async def clear_bed(
        printer_id: int,
        action_token: str,
        success: bool,
        rating: Optional[int] = None,
    ):
        headers = {
            "X-Action-Token": action_token,
        }

        # Hacky: extract company_id from action_token itself (jwt)
        company_id = json.loads(
            base64.b64decode(action_token.split(".")[1] + "===").decode("utf-8")
        )["company"]

        endpoint = (
            SimplyPrintURL().api_url
            / str(company_id)
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
                    raise Exception(f"Failed to clear bed: {await response.text()}")

                return await response.json()

    @staticmethod
    async def start_next_print(printer_id: int, action_token: str):
        headers = {
            "X-Action-Token": action_token,
        }

        # Hacky: extract company_id from action_token itself (jwt)
        company_id = json.loads(
            base64.b64decode(action_token.split(".")[1] + "===").decode("utf-8")
        )["company"]

        data = {
            "pid": printer_id,
            "next_queue_item": True,
        }

        endpoint = (
            SimplyPrintURL().api_url
            / str(company_id)
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
                    raise Exception(
                        f"Failed to start next print: {await response.text()}"
                    )

                return await response.json()
