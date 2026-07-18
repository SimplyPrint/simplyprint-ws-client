"""The identity seam every add path funnels through: slot id + hardware-match id.

Two separate identities are assigned once here, in the open, rather than buried
in a web handler:

* ``unique_id`` -- the *slot* reference: a stable UUID, the ``client_list`` key
  and the backend handle. Assigned once and never re-keyed, so it survives an IP
  change *and* a physical-printer swap.
* a stable *hardware* id for matching a re-discovered device back to its slot --
  returned directly by ``config.hardware_identity()`` (serial / board id / guid /
  MAC). This is what discovery correlates on; it never becomes the ``unique_id``.

Each config returns its address values through
:meth:`~simplyprint_ws_client.core.config.PrinterConfig.network_addresses`.
"""

from __future__ import annotations

import uuid


def capture_hardware_id(config) -> None:
    """Capture a stable hardware id for matching when the brand exposes none.

    Brands with a serial/guid already answer ``hardware_identity``; for the rest
    (a bare subnet-scanned printer) resolve the device's MAC from its address and
    store it in ``config.mac``, so a re-discovery on a new IP still finds this
    slot. A no-op once a hardware id (brand id or a previously-captured MAC) is
    present.
    """
    from simplyprint_ws_client.integration.discovery.reconcile import DeviceReconciler

    if DeviceReconciler.normalize_identity(config.hardware_identity()) is not None:
        return
    address = config.primary_network_address()
    if not address:
        return
    from simplyprint_ws_client.integration.discovery.mac import resolve_mac

    host = DeviceReconciler.normalize_address(address)
    if host is not None:
        config.mac = resolve_mac(host)


def assign_unique_id(config) -> str:
    """Assign the slot's stable ``unique_id`` (once, never re-keyed) and capture a
    hardware-match id -- the one visible seam every add path funnels through."""
    capture_hardware_id(config)
    config.unique_id = config.unique_id or str(uuid.uuid4())
    return config.unique_id
