from __future__ import annotations

import logging
import subprocess
import sys
from typing import Any

_LOGGER = logging.getLogger("system.command")
_OUTPUT_LIMIT = 4000


def hidden_windows_subprocess_kwargs() -> dict[str, Any]:
    """Hide console windows for background child processes on Windows."""
    if sys.platform != "win32":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0  # SW_HIDE
    return {
        "creationflags": subprocess.CREATE_NO_WINDOW,
        "startupinfo": startupinfo,
    }


def _command_text(args: Any) -> str:
    if isinstance(args, (list, tuple)):
        return subprocess.list2cmdline([str(part) for part in args])
    return str(args)


def _with_hidden_window(kwargs: dict[str, Any]) -> dict[str, Any]:
    merged = dict(kwargs)
    if sys.platform != "win32":
        return merged

    startupinfo = merged.get("startupinfo") or subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = 0  # SW_HIDE
    merged["startupinfo"] = startupinfo
    merged["creationflags"] = (
        int(merged.get("creationflags") or 0) | subprocess.CREATE_NO_WINDOW
    )
    return merged


def _log_output(logger: logging.Logger, stream: str, output: Any) -> None:
    if not output:
        return
    if isinstance(output, bytes):
        output = output.decode(errors="replace")
    text = str(output).strip()
    if not text:
        return
    if len(text) > _OUTPUT_LIMIT:
        text = text[:_OUTPUT_LIMIT] + "..."
    logger.debug("system command %s: %s", stream, text)


def run(
    args: Any,
    *,
    action: str | None = None,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess:
    logger = logger or _LOGGER
    command = _command_text(args)
    label = action or command
    logger.debug("system command start: %s", label)
    try:
        completed = subprocess.run(args, **_with_hidden_window(kwargs))
    except (OSError, subprocess.SubprocessError):
        logger.warning("system command failed to start: %s", label, exc_info=True)
        raise
    logger.debug("system command exit: %s rc=%s", label, completed.returncode)
    _log_output(logger, "stdout", getattr(completed, "stdout", None))
    _log_output(logger, "stderr", getattr(completed, "stderr", None))
    return completed


def check_output(
    args: Any,
    *,
    action: str | None = None,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> Any:
    logger = logger or _LOGGER
    command = _command_text(args)
    label = action or command
    logger.debug("system command start: %s", label)
    try:
        output = subprocess.check_output(args, **_with_hidden_window(kwargs))
    except (OSError, subprocess.SubprocessError):
        logger.warning("system command failed: %s", label, exc_info=True)
        raise
    logger.debug("system command exit: %s rc=0", label)
    _log_output(logger, "stdout", output)
    return output


def popen(
    args: Any,
    *,
    action: str | None = None,
    logger: logging.Logger | None = None,
    **kwargs: Any,
) -> subprocess.Popen:
    logger = logger or _LOGGER
    command = _command_text(args)
    label = action or command
    logger.debug("system command spawn: %s", label)
    try:
        process = subprocess.Popen(args, **_with_hidden_window(kwargs))
    except OSError:
        logger.warning("system command failed to spawn: %s", label, exc_info=True)
        raise
    logger.debug("system command spawned: %s pid=%s", label, process.pid)
    return process
