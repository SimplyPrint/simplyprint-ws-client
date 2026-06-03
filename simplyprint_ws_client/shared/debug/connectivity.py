__all__ = ["ConnectivityReport"]

import asyncio
import datetime
import logging
import queue
import socket
import threading
from pathlib import Path
from typing import List, Optional, Dict

import aiohttp
import netifaces
import psutil
from aiohttp import ClientWSTimeout
from pydantic import BaseModel, Field

from ..sp.url_builder import SimplyPrintBackend


class WebSocketTestResult(BaseModel):
    url: str
    success: bool
    error_message: Optional[str] = None
    response_headers: Optional[Dict[str, str]] = None
    latency_ms: Optional[float] = None


class DNSResolutionResult(BaseModel):
    host: str
    resolved_ips: List[str]
    success: bool
    error_message: Optional[str] = None


class HTTPTestResult(BaseModel):
    url: str
    success: bool
    status_code: Optional[int] = None
    response_snippet: Optional[str] = None
    error_message: Optional[str] = None


class NetworkInterface(BaseModel):
    name: str
    addresses: List[str]
    is_up: bool


class LocalNetworkInfo(BaseModel):
    hostname: str
    network_interfaces: List[NetworkInterface]


class ConnectivityReport(BaseModel):
    timestamp: datetime.datetime = Field(
        default_factory=lambda: ConnectivityReport.utc_now()
    )
    dns_results: List[DNSResolutionResult] = []
    websocket_results: List[WebSocketTestResult] = []
    http_results: List[HTTPTestResult] = []
    local_network_info: LocalNetworkInfo

    @staticmethod
    def utc_now() -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)

    @staticmethod
    def get_local_network_info() -> LocalNetworkInfo:
        local_hostname = socket.gethostname()

        interfaces = []
        for interface in netifaces.interfaces():
            addrs = []
            iface_addrs = netifaces.ifaddresses(interface)
            for family in (netifaces.AF_INET, netifaces.AF_INET6):
                if family in iface_addrs:
                    addrs.extend(addr_info["addr"] for addr_info in iface_addrs[family])

            iface_stats = psutil.net_if_stats().get(interface)
            is_up = iface_stats.isup if iface_stats else False

            interfaces.append(
                NetworkInterface(name=interface, addresses=addrs, is_up=is_up)
            )

        return LocalNetworkInfo(hostname=local_hostname, network_interfaces=interfaces)

    @staticmethod
    def resolve_dns(host: str, logger: logging.Logger) -> DNSResolutionResult:
        logger.info(f"Resolving DNS for host: {host}")
        try:
            resolved_ips = list(
                set(res[4][0] for res in socket.getaddrinfo(host, None))
            )
            logger.info(f"Resolved {host} to IPs: {resolved_ips}")
            return DNSResolutionResult(
                host=host, resolved_ips=resolved_ips, success=True
            )
        except socket.gaierror as e:
            logger.error(f"Failed DNS resolution for {host}: {e}")
            return DNSResolutionResult(
                host=host, resolved_ips=[], success=False, error_message=str(e)
            )

    @staticmethod
    def websocket_test(
        url: str, websocket_timeout: int, logger: logging.Logger
    ) -> WebSocketTestResult:
        result_queue = queue.Queue()

        if ClientWSTimeout is not None:
            websocket_timeout = ClientWSTimeout(ws_close=websocket_timeout)

        def test_ws():
            async def inner():
                async with aiohttp.ClientSession() as session:
                    start_time = ConnectivityReport.utc_now()
                    try:
                        async with session.ws_connect(
                            url, timeout=websocket_timeout
                        ) as ws:
                            latency = (
                                ConnectivityReport.utc_now() - start_time
                            ).total_seconds() * 1000
                            logger.info(
                                f"Websocket connection successful to {url} (latency: {latency:.2f}ms)"
                            )
                            return WebSocketTestResult(
                                url=url,
                                success=True,
                                response_headers=dict(ws._response.headers),
                                latency_ms=latency,
                            )
                    except Exception as e:
                        logger.error(f"Websocket connection failed to {url}: {e}")

                        return WebSocketTestResult(
                            url=url, success=False, error_message=str(e)
                        )

            try:
                result_queue.put(asyncio.run(inner()))
            finally:
                result_queue.put(None)

        thread = threading.Thread(target=test_ws)
        thread.start()
        thread.join()
        return result_queue.get()

    @staticmethod
    def http_test(url: str, timeout: int, logger: logging.Logger) -> HTTPTestResult:
        result_queue = queue.Queue()

        def test_http():
            async def inner():
                async with aiohttp.ClientSession() as session:
                    try:
                        async with session.get(url, timeout=timeout) as response:
                            text = await response.text()
                            snippet = text[:100]
                            logger.info(
                                f"HTTP request successful to {url} with status {response.status}"
                            )
                            return HTTPTestResult(
                                url=url,
                                success=True,
                                status_code=response.status,
                                response_snippet=snippet,
                            )
                    except Exception as e:
                        logger.error(f"HTTP request failed to {url}: {e}")
                        return HTTPTestResult(
                            url=url, success=False, error_message=str(e)
                        )

            try:
                result_queue.put(asyncio.run(inner()))
            finally:
                result_queue.put(None)

        thread = threading.Thread(target=test_http)
        thread.start()
        thread.join()
        return result_queue.get()

    @staticmethod
    def generate(
        urls: List[str],
        http_urls: List[str],
        additional_dns_tests: List[str],
        logger: logging.Logger = logging.getLogger("connection_debugger"),
        websocket_timeout: int = 10,
        http_timeout: int = 10,
    ) -> "ConnectivityReport":
        logger.info("Beginning connectivity test suite")

        local_info = ConnectivityReport.get_local_network_info()
        report = ConnectivityReport(local_network_info=local_info)

        dns_hosts_to_test = list(
            set(
                url.replace("ws://", "").replace("wss://", "").split("/")[0]
                for url in urls
            )
            | set(additional_dns_tests)
        )
        for host in dns_hosts_to_test:
            dns_result = ConnectivityReport.resolve_dns(host, logger)
            report.dns_results.append(dns_result)

        for url in urls:
            websocket_result = ConnectivityReport.websocket_test(
                url, websocket_timeout, logger
            )
            report.websocket_results.append(websocket_result)

        for url in http_urls:
            http_result = ConnectivityReport.http_test(url, http_timeout, logger)
            report.http_results.append(http_result)

        return report

    @staticmethod
    def generate_default(**kwargs) -> "ConnectivityReport":
        return ConnectivityReport.generate(
            [
                str(SimplyPrintBackend.PRODUCTION.urls().ws),
                str(SimplyPrintBackend.STAGING.urls().ws),
                str(SimplyPrintBackend.TESTING.urls().ws),
            ],
            [
                str(SimplyPrintBackend.PRODUCTION.urls().api),
                str(SimplyPrintBackend.STAGING.urls().api),
                str(SimplyPrintBackend.TESTING.urls().api),
            ],
            ["1.1.1.1", "google.com"],
            **kwargs,
        )

    @staticmethod
    def read_previous_reports(path: Path) -> List["ConnectivityReport"]:
        report_files = sorted(path.glob("connectivity_report_*.json"), reverse=True)
        return [
            ConnectivityReport.model_validate_json(f.read_text()) for f in report_files
        ]

    def summary(self) -> str:
        """Returns a string with a count of all tests, and how many failed."""
        summary_str = f"DNS tests: {len(self.dns_results)}\n"
        failed_dns = sum(1 for res in self.dns_results if not res.success)
        summary_str += f"Failed DNS tests: {failed_dns}\n"
        summary_str += f"Websocket tests: {len(self.websocket_results)}\n"
        failed_ws = sum(1 for res in self.websocket_results if not res.success)
        summary_str += f"Failed Websocket tests: {failed_ws}\n"
        summary_str += f"HTTP tests: {len(self.http_results)}\n"
        failed_http = sum(1 for res in self.http_results if not res.success)
        summary_str += f"Failed HTTP tests: {failed_http}\n"
        return summary_str

    def store_in_path(self, path: Path, max_reports: int = 5) -> Path:
        path.mkdir(parents=True, exist_ok=True)

        report_files = sorted(path.glob("connectivity_report_*.json"), reverse=True)
        for old_file in report_files[max_reports - 1 :]:
            old_file.unlink()

        filename = (
            f"connectivity_report_{self.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        )
        full_path = path / filename

        with open(full_path, "w") as f:
            f.write(self.model_dump_json(indent=4))

        return full_path
