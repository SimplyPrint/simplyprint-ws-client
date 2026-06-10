"""The one builder of the addable-candidate surface.

Collects everything the user could add right now -- LAN-discovered devices plus
every linked cloud account's devices -- then runs a single generalized dedupe
pass over the merged list:

1. **Drop already-added.** A candidate whose hardware identity matches a stored
   config for its brand is removed (it is already a printer).
2. **Collapse duplicates.** Candidates sharing ``(type, hardware identity)``
   collapse to one card; when a LAN and a cloud candidate are the same printer
   the LAN one wins (it carries the host the wizard needs).
3. **Fail open.** A candidate with no resolvable hardware identity is never
   dropped or collapsed -- we never hide a printer we can't positively prove is
   already added or a duplicate.

Identity is computed by :mod:`reconcile`, the one hardware-identity owner, so the
already-configured filter and add-time de-duplication share a single rule.

The app supplies its linked cloud accounts as ``account_providers`` (a mapping of
client-type key to provider instance); the library owns no provider registry.
"""

from __future__ import annotations

from typing import Mapping, Optional

from simplyprint_ws_client.integration.discovery.reconcile import (
    DeviceReconciler,
    device_hardware_id,
)


def _lan_candidate(result) -> dict:
    """A LAN discovery result as a neutral candidate. ``model`` is the brand's
    human-readable model name (``model_name``) for display; ``device_type`` is the
    raw value the add flow seeds; the brand may also surface a ``device_image_url``
    it has already resolved -- all kept opaque here."""
    extra = dict(result.extra or {})
    device_type = extra.get("device_type")
    model = extra.get("model_name") or device_type
    model_image_url = extra.get("device_image_url")
    return {
        "source": "lan",
        "type": result.type,
        "host": result.host,
        "serial": result.serial,
        "name": result.name,
        "model": str(model) if model else "",
        "device_type": str(device_type) if device_type else None,
        "model_image_url": str(model_image_url) if model_image_url else None,
        "online": None,
        "account_uid": None,
        "extra": extra,
    }


def _cloud_candidate(provider: str, uid: str, device) -> dict:
    """A cloud-account device as a neutral candidate. The provider is the client
    type; the account ``uid`` rides along so the add endpoint can adopt it."""
    return {
        "source": "cloud",
        "type": provider,
        "host": None,
        "serial": device.serial,
        "name": device.name,
        "model": device.model,
        "model_image_url": device.model_image_url,
        "online": bool(device.online),
        "account_uid": uid,
        "extra": {},
    }


async def _cloud_candidates(account_providers: Mapping[str, object]) -> "list[dict]":
    """Every device every linked account (across providers) knows about, as cloud
    candidates. Best-effort per account: a provider that fails to list (auth gone,
    cloud down) is skipped so one bad account can't blank the whole list."""
    out: "list[dict]" = []
    for key in sorted(account_providers):
        instance = account_providers[key]
        if instance is None:
            continue
        for account in instance.get_accounts():
            try:
                devices = await instance.get_devices(account.uid)
            except Exception:
                continue
            for device in devices or []:
                out.append(_cloud_candidate(key, account.uid, device))
    return out


def _hardware_id(candidate: dict) -> Optional[str]:
    return device_hardware_id(
        candidate.get("host"), candidate.get("serial"), candidate.get("extra")
    )


def _dedupe(client_app, candidates: "list[dict]") -> "list[dict]":
    """Drop already-added candidates and collapse same-printer duplicates.

    LAN candidates are considered before cloud ones so a LAN/cloud collision keeps
    the LAN card (it carries the host the wizard needs). A candidate with no
    hardware identity is kept verbatim -- never matched against a config, never
    folded into another card.
    """
    ordered = [c for c in candidates if c["source"] == "lan"]
    ordered += [c for c in candidates if c["source"] != "lan"]

    out: "list[dict]" = []
    seen: "set[tuple[str, str]]" = set()
    for candidate in ordered:
        hardware_id = _hardware_id(candidate)
        if hardware_id is None:
            out.append(candidate)  # fail open: unidentifiable, never hidden
            continue

        manager = client_app.get_config_manager(client_key=candidate["type"])
        if DeviceReconciler(manager).matching(
            hardware_id=hardware_id, host=candidate.get("host")
        ):
            continue  # already a stored printer

        key = (candidate["type"], hardware_id)
        if key in seen:
            continue  # same printer already kept (LAN wins by ordering)
        seen.add(key)
        out.append(candidate)
    return out


async def build_candidates(
    client_app,
    discovery_store,
    *,
    account_providers: Optional[Mapping[str, object]] = None,
) -> "list[dict]":
    """The deduped addable-candidate list for the discovery/add surface."""
    candidates = [_lan_candidate(result) for result in discovery_store.current()]
    candidates.extend(await _cloud_candidates(account_providers or {}))
    return _dedupe(client_app, candidates)
