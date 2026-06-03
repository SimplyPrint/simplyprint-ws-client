"""Reusable discovery primitives."""

from simplyprint_ws_client.contrib.discovery.device import DiscoveredDevice
from simplyprint_ws_client.contrib.discovery.ssdp import SSDPRequest, SSDPRequestParser

__all__ = ["DiscoveredDevice", "SSDPRequest", "SSDPRequestParser"]
