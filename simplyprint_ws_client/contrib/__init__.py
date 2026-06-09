"""Reusable library components an integration composes.

``contrib`` is the home for the library's building blocks -- the
:mod:`~simplyprint_ws_client.contrib.connection` subsystem (pooled, self-healing
connections with the per-protocol ``mqtt`` / ``ws`` front doors), the guided
:mod:`~simplyprint_ws_client.contrib.flow` engine, and the
:mod:`~simplyprint_ws_client.contrib.logging` facility. Import the submodule you
need (e.g. ``from simplyprint_ws_client.contrib.connection import ws``); the common
names are also re-exported from the package root ``simplyprint_ws_client``.

This package ``__init__`` is intentionally import-free so that low-level
submodules (``conn``) can be imported by ``core`` without dragging in the
higher-level ones (e.g. ``flow``).
"""
