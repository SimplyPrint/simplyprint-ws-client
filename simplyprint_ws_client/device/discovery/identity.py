"""The identity seam every add path funnels through: slot id + hardware-match id.

Two separate identities are assigned once here, in the open, rather than buried
in a web handler:

* ``unique_id`` -- the *slot* reference: a stable UUID, the ``client_list`` key
  and the backend handle. Assigned once and never re-keyed, so it survives an IP
  change *and* a physical-printer swap.
* a stable *hardware* id for matching a re-discovered device back to its slot --
  the brand's ``stable_hardware_id`` (serial / board id / guid) or, when the brand
  exposes none, the device's MAC resolved from the LAN and stored in
  ``config.mac``. This is what discovery correlates on; it never becomes the
  ``unique_id``.

Which config fields may carry the device's address is the config class's
business: :attr:`~simplyprint_ws_client.core.config.PrinterConfig.network_address_fields`.
"""

from __future__ import annotations

import uuid
from typing import Optional


def _config_host(config) -> Optional[str]:
    for field in type(config).network_address_fields:
        value = getattr(config, field, None)
        if value:
            return str(value)
    return None


def capture_hardware_id(config) -> None:
    """Capture a stable hardware id for matching when the brand exposes none.

    Brands with a serial/guid already answer ``stable_hardware_id``; for the rest
    (a bare subnet-scanned printer) resolve the device's MAC from its address and
    store it in ``config.mac``, so a re-discovery on a new IP still finds this
    slot. A no-op once a hardware id (brand id or a previously-captured MAC) is
    present.
    """
    if config.stable_hardware_id() or getattr(config, "mac", None):
        return
    host = _config_host(config)
    if not host:
        return
    from simplyprint_ws_client.device.discovery.mac import resolve_mac

    config.mac = resolve_mac(host)


def assign_unique_id(config) -> str:
    """Assign the slot's stable ``unique_id`` (once, never re-keyed) and capture a
    hardware-match id -- the one visible seam every add path funnels through."""
    capture_hardware_id(config)
    config.unique_id = config.unique_id or str(uuid.uuid4())
    return config.unique_id
