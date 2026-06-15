import logging

import pytest

from simplyprint_ws_client.common import process


class _FakeStartupInfo:
    def __init__(self):
        self.dwFlags = 0
        self.wShowWindow = None


def _patch_windows_subprocess(monkeypatch):
    monkeypatch.setattr(process.sys, "platform", "win32")
    monkeypatch.setattr(
        process.subprocess, "STARTUPINFO", _FakeStartupInfo, raising=False
    )
    monkeypatch.setattr(
        process.subprocess, "STARTF_USESHOWWINDOW", 1, raising=False
    )
    monkeypatch.setattr(
        process.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False
    )


def test_hidden_windows_subprocess_kwargs_noops_off_windows(monkeypatch):
    monkeypatch.setattr(process.sys, "platform", "linux")

    assert process.hidden_windows_subprocess_kwargs() == {}


def test_hidden_windows_subprocess_kwargs_hides_window(monkeypatch):
    _patch_windows_subprocess(monkeypatch)

    kwargs = process.hidden_windows_subprocess_kwargs()

    assert kwargs["creationflags"] == 0x08000000
    assert kwargs["startupinfo"].dwFlags == 1
    assert kwargs["startupinfo"].wShowWindow == 0


def test_run_hides_windows_console_and_logs(monkeypatch, caplog):
    _patch_windows_subprocess(monkeypatch)
    captured = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return process.subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(process.subprocess, "run", fake_run)
    caplog.set_level("DEBUG", logger="system.command")

    completed = process.run(["tool", "arg"], action="test command", capture_output=True)

    assert completed.returncode == 0
    assert captured["kwargs"]["creationflags"] == 0x08000000
    assert captured["kwargs"]["startupinfo"].wShowWindow == 0
    assert "system command start: test command" in caplog.text
    assert "system command exit: test command rc=0" in caplog.text
    assert "system command stdout: ok" in caplog.text


def test_check_output_hides_windows_console(monkeypatch):
    _patch_windows_subprocess(monkeypatch)
    captured = {}

    def fake_check_output(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return b"ok"

    monkeypatch.setattr(process.subprocess, "check_output", fake_check_output)

    assert process.check_output(["tool"], action="test output") == b"ok"
    assert captured["kwargs"]["creationflags"] == 0x08000000


def test_check_output_missing_executable_logs_debug_not_warning(monkeypatch, caplog):
    # A missing optional host tool (e.g. iwgetid) must not spam a WARNING +
    # traceback at startup -- it's expected; log it at debug and re-raise so the
    # caller can degrade gracefully.
    def fake_check_output(args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "iwgetid")

    monkeypatch.setattr(process.subprocess, "check_output", fake_check_output)
    caplog.set_level(logging.DEBUG, logger="system.command")

    with pytest.raises(FileNotFoundError):
        process.check_output(["iwgetid", "-r"], action="read host hardware info")

    assert "system command unavailable: read host hardware info" in caplog.text
    # No warning, and no traceback was logged.
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert "Traceback" not in caplog.text


def test_popen_hides_windows_console_and_logs(monkeypatch, caplog):
    _patch_windows_subprocess(monkeypatch)
    captured = {}

    class FakeProcess:
        pid = 123

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(process.subprocess, "Popen", fake_popen)
    caplog.set_level("DEBUG", logger="system.command")

    spawned = process.popen(["tool"], action="test spawn")

    assert spawned.pid == 123
    assert captured["kwargs"]["creationflags"] == 0x08000000
    assert "system command spawn: test spawn" in caplog.text
    assert "system command spawned: test spawn pid=123" in caplog.text
