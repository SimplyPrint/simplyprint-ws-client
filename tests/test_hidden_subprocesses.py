import subprocess

import pytest

from simplyprint_ws_client.common.hardware import physical_machine
from simplyprint_ws_client.integration.discovery import mac


@pytest.fixture(autouse=True)
def clear_mac_caches():
    mac._mac_cache.clear()
    mac._mac_negative_cache.clear()
    yield
    mac._mac_cache.clear()
    mac._mac_negative_cache.clear()


def test_callonce_caches_none_failures():
    calls = []

    @physical_machine.callonce
    def probe():
        calls.append(True)
        return None

    assert probe() is None
    assert probe() is None
    assert calls == [True]


def test_physical_machine_check_output_uses_system_command_wrapper(monkeypatch):
    captured = {}

    def fake_check_output(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return b"Name\r\nCPU\r\n"

    monkeypatch.setattr(physical_machine, "system_check_output", fake_check_output)

    physical_machine.capped_check_output(["wmic", "cpu", "get", "name"])

    assert captured["args"] == (["wmic", "cpu", "get", "name"],)
    assert captured["kwargs"]["action"] == "read host hardware info"
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["timeout"] == 1.0


def test_macos_ssid_uses_networksetup_when_airport_is_missing(monkeypatch):
    calls = []

    def fake_exists(path):
        return path == physical_machine._MACOS_NETWORKSETUP_PATH

    def fake_check_output(argv):
        calls.append(argv)
        if argv == [physical_machine._MACOS_NETWORKSETUP_PATH, "-listallhardwareports"]:
            return (
                b"Hardware Port: Ethernet\n"
                b"Device: en7\n\n"
                b"Hardware Port: Wi-Fi\n"
                b"Device: en0\n"
                b"Ethernet Address: aa:bb:cc:dd:ee:ff\n"
            )
        if argv == [
            physical_machine._MACOS_NETWORKSETUP_PATH,
            "-getairportnetwork",
            "en0",
        ]:
            return b"Current Wi-Fi Network: Shop Floor\n"
        raise AssertionError(f"unexpected command: {argv}")

    monkeypatch.setattr(physical_machine.os.path, "exists", fake_exists)
    monkeypatch.setattr(physical_machine, "capped_check_output", fake_check_output)

    ssid_macos = physical_machine.PhysicalMachine._PhysicalMachine__ssid_macos.__wrapped__

    assert ssid_macos() == "Shop Floor"
    assert all(physical_machine._MACOS_AIRPORT_PATH not in call for call in calls)


def test_discovery_mac_command_uses_system_command_wrapper(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="")

    monkeypatch.setattr(mac.sys, "platform", "linux")
    monkeypatch.setattr(mac, "run_command", fake_run)

    assert mac._mac_from_command("192.168.1.50", timeout=0.1) is None
    assert len(calls) == 1
    assert calls[0][0] == ["ip", "neigh", "show", "192.168.1.50"]
    assert calls[0][1]["action"] == "resolve MAC from OS neighbour table"


def test_discovery_mac_command_does_not_try_ip_on_windows(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(
            argv,
            0,
            stdout="192.168.1.50 aa-bb-cc-dd-ee-ff dynamic",
        )

    monkeypatch.setattr(mac.sys, "platform", "win32")
    monkeypatch.setattr(mac, "run_command", fake_run)

    assert mac._mac_from_command("192.168.1.50", timeout=0.1) == "aa:bb:cc:dd:ee:ff"
    assert calls == [["arp", "-a", "192.168.1.50"]]


def test_mac_from_proc_does_not_touch_proc_off_linux(monkeypatch):
    def fail_open(*args, **kwargs):
        raise AssertionError("/proc should not be read off Linux")

    monkeypatch.setattr(mac.sys, "platform", "win32")
    monkeypatch.setattr("builtins.open", fail_open)

    assert mac._mac_from_proc("192.168.1.50") is None


def test_neighbour_commands_are_platform_specific(monkeypatch):
    monkeypatch.setattr(mac.sys, "platform", "win32")
    assert mac._neighbour_command("192.168.1.50") == ["arp", "-a", "192.168.1.50"]

    monkeypatch.setattr(mac.sys, "platform", "linux")
    assert mac._neighbour_command("192.168.1.50") == [
        "ip",
        "neigh",
        "show",
        "192.168.1.50",
    ]

    monkeypatch.setattr(mac.sys, "platform", "darwin")
    assert mac._neighbour_command("192.168.1.50") == ["arp", "-n", "192.168.1.50"]


def test_resolve_mac_uses_longer_positive_cache(monkeypatch):
    calls = []

    def fake_mac_from_proc(ip):
        calls.append(ip)
        return "aa:bb:cc:dd:ee:ff"

    monkeypatch.setattr(mac, "_resolve_host_ip", lambda host: "192.168.1.50")
    monkeypatch.setattr(mac, "_mac_from_proc", fake_mac_from_proc)
    monkeypatch.setattr(mac, "_mac_from_command", lambda ip, timeout: None)
    monkeypatch.setattr(mac, "_warm_neighbour_table", lambda ip, timeout: None)

    assert mac.resolve_mac("printer.local") == "aa:bb:cc:dd:ee:ff"
    assert mac.resolve_mac("printer.local") == "aa:bb:cc:dd:ee:ff"
    assert calls == ["192.168.1.50"]
    assert mac._MAC_CACHE_TTL == 15 * 60


def test_resolve_mac_caches_negative_results(monkeypatch):
    calls = []

    def fake_mac_from_proc(ip):
        calls.append(("proc", ip))
        return None

    def fake_mac_from_command(ip, timeout):
        calls.append(("command", ip))
        return None

    def fake_warm(ip, timeout):
        calls.append(("warm", ip))

    monkeypatch.setattr(mac, "_resolve_host_ip", lambda host: "192.168.1.50")
    monkeypatch.setattr(mac, "_mac_from_proc", fake_mac_from_proc)
    monkeypatch.setattr(mac, "_mac_from_command", fake_mac_from_command)
    monkeypatch.setattr(mac, "_warm_neighbour_table", fake_warm)

    assert mac.resolve_mac("printer.local") is None
    assert mac.resolve_mac("printer.local") is None
    assert calls == [
        ("proc", "192.168.1.50"),
        ("warm", "192.168.1.50"),
        ("proc", "192.168.1.50"),
        ("command", "192.168.1.50"),
    ]
    assert mac._MAC_NEGATIVE_CACHE_TTL == 5 * 60


@pytest.mark.parametrize(
    ("system", "expected", "action"),
    [
        ("Linux", ["sudo", "reboot"], "restart Linux host"),
        ("Darwin", ["reboot"], "restart macOS host"),
        ("Windows", ["shutdown", "/r", "/t", "1"], "restart Windows host"),
    ],
)
def test_physical_machine_restart_uses_system_command_wrapper(
    monkeypatch, system, expected, action
):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))

    monkeypatch.setattr(physical_machine.platform, "system", lambda: system)
    monkeypatch.setattr(physical_machine, "run_command", fake_run)

    physical_machine.PhysicalMachine.restart()

    assert calls == [(expected, {"action": action})]


@pytest.mark.parametrize(
    ("system", "expected", "action"),
    [
        ("Linux", ["sudo", "shutdown", "now"], "shut down Linux host"),
        ("Darwin", ["shutdown", "now"], "shut down macOS host"),
        ("Windows", ["shutdown", "/s", "/t", "1"], "shut down Windows host"),
    ],
)
def test_physical_machine_shutdown_uses_system_command_wrapper(
    monkeypatch, system, expected, action
):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))

    monkeypatch.setattr(physical_machine.platform, "system", lambda: system)
    monkeypatch.setattr(physical_machine, "run_command", fake_run)

    physical_machine.PhysicalMachine.shutdown()

    assert calls == [(expected, {"action": action})]
