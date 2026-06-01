# simplyprint-ws-client — Claude operating rules

This is the shared runtime every SimplyPrint printer integration builds
on. Seven integrations consume it; six pin older release-candidate
versions and adopt changes on their own schedule, so breaking changes here
are acceptable when the work calls for them.

## What this library is

- **`core/`** — the SimplyPrint-backend engine: the WebSocket protocol
  (`ws_protocol/`), the `PrinterState` model + change tracking (`state/`),
  the `Client`/`consume()` message pipeline, config storage, app
  scaffolding. This is the cloud side.
- **`contrib/`** — the reusable, composable building blocks an integration
  subclasses to talk to a *printer*: `PrinterClient` (lifecycle + status
  reduction), `FileTransfer` (prepare → download → transform → upload →
  await-firmware), `onboard_printer` (discover → verify → setup), the
  swappable WebSocket `transport`, the threaded `connection` pool, the
  reactive/diffable `model`, and the `logging` facility.
- **`shared/`** — leaf primitives both layers use (asyncio helpers,
  `Backoff`, `BoundedVariable`, `Synchronized`/`Stoppable`, file download,
  hardware snapshot, SimplyPrint HTTP API, CLI).

## The contrib boundary

`contrib/` is the home for anything brand-agnostic that "could live in any
integration." It must contain **zero** brand names, brand enums, port
numbers, topic shapes, or device-model field names — the same hard rule
the integration applies, enforced here. A brand difference is expressed as
a hook (abstract method, class attribute, callback), never as a branch on
brand identity. If a difference cannot be expressed as a clean hook, the
abstraction is wrong — fix its shape, or accept that the two things are
genuinely not the same and leave the brand-specific one in the
integration. KISS sits above DRY: sharing a name is not a goal, sharing
real behavior is.

## The import-cycle rule

`contrib/__init__.py` is intentionally **import-free**. `core` (and any
contrib *leaf* module) may import contrib leaves — `transport`,
`connection`, `model`, `transfer`, `onboarding`, `logging` — but nothing
in `core` or a contrib leaf may import `contrib/printer_client.py`, which
itself imports `core`. New leaves must preserve this DAG. Optional/heavy
third-party impls (paho-mqtt, websocket-client, aiohttp) are served lazily
via PEP 562 `__getattr__` in the relevant package `__init__`, so a plain
`import simplyprint_ws_client.contrib.connection` never drags them in;
keep them optional extras, never hard dependencies.

## Version floor: Python 3.9

`requires-python = ">=3.9"`. In runtime code: **no `match`**, **no PEP 604
`X | Y` unions** evaluated at runtime, **no bare `typing.Self`**. Use
`typing.Optional`/`Union`/`List`/`Dict`, add `from __future__ import
annotations` to stringize annotations, and for `Self` use the established
pattern (`typing-extensions` is already a hard dependency):

```python
try:
    from typing import Self
except ImportError:
    from typing_extensions import Self
```

The integration repos run 3.10+ and keep their syntax; code **moved into
this library is rewritten to the 3.9 floor**.

## Lint / format / green floors

- `ruff check --target-version py39` and `ruff format --check` must be
  clean. (No `[tool.ruff]` block exists yet — pass `--target-version` on
  the CLI; adding `target-version = "py39"` to `pyproject.toml` is a fine
  follow-up so the floor is enforced by default.)
- `.venv/bin/python -m pytest -q` → **118 passed** is the green floor.
- Use the `.venv` directly. **Do NOT use `uv run`/`uv sync`** — it would
  re-pin the editable install and break the integration's live source
  link.

## Migrations from integrations are hard-cut

Code promoted from an integration lands here fully rewritten to the 3.9
floor; the integration deletes its copy and repoints every import at the
library. No re-export shims in either direction. Make behavior
byte-perfect first (the highest-risk contract is `Client.consume()`'s
message order/dedup/`reset` — guard it with a golden-equivalence test
before any change to the state engine), then collapse duplication, then
make the hierarchy read cleanly. Prove the abstraction on the hardest
brand first.

## Tests are a contract

Never weaken, skip, xfail, comment-out, or delete a test to make the suite
pass. When a test pins an internal you deliberately changed, update it to
the new contract with its assertions unchanged, and say so. New shared
abstractions ship with brand-free tests that exercise them without any
integration present.
