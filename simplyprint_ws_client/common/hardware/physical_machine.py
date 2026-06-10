import functools
import os
import platform
import re
import socket
import subprocess
from typing import Optional

import netifaces
import psutil

from simplyprint_ws_client.common.utils.exception_as_value import exception_as_value


def callonce(func):
    result = None

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal result

        if result is None:
            result = func(*args, **kwargs)

        return result

    return wrapper


# FIXME:
# There is a bug in Python that ends up with process.communicate hanging even after
# The process it runs has exited. This is a workaround for that issue by just introducing a timeout
# since none of the commands we call do anything important that needs time to run.
# To reproduce the issue spawning MultiProcessing processes while calling check_output / run
# seems to do the trick.
capped_check_output = functools.partial(
    subprocess.check_output, shell=False, timeout=1.0
)


class PhysicalMachine:
    """
    Provides hardware and platform information, abstractly.
    """

    @staticmethod
    def get_usage():
        temperature: float = 0

        try:
            temperatures = psutil.sensors_temperatures()

            # Find the first temperature sensor that is not None
            # Priority is given in reverse order
            temperature_keys = {
                "coretemp",
                "cpu-thermal",
                "cpu_thermal",
                "soc_thermal",
            } & set(temperatures.keys())

            if len(temperature_keys) > 0:
                temperature_key = temperature_keys.pop()
                temperature = (
                    temperatures[temperature_key][0].current
                    if temperatures[temperature_key][0].current is not None
                    else 0
                )
        except AttributeError:
            pass

        return {
            "usage": round(psutil.cpu_percent()),
            "temp": round(temperature),
            "memory": round(psutil.virtual_memory().percent),
            "flags": 0,
        }

    @classmethod
    def get_info(cls):
        return {
            "python_version": cls.python_version(),
            "machine": cls.machine(),
            "os": platform.system(),
            "mac": cls.mac_address(),
            "is_ethernet": cls.is_ethernet(),
            "ssid": cls.ssid(),
            "hostname": cls.hostname(),
            "local_ip": cls.local_ip(),
            "core_count": cls.core_count(),
            "total_memory": cls.total_memory(),
        }

    @staticmethod
    def python_version() -> str:
        """
        Returns the Python version.
        """
        return platform.python_version()

    @staticmethod
    @callonce
    @exception_as_value(return_default=True)
    def __get_cpu_model_linux() -> Optional[str]:
        info_path = "/proc/cpuinfo"

        with open(info_path, "r") as f:
            data = f.read()

        cpu_items = [item.strip() for item in data.split("\n\n") if item.strip()]

        match = re.search(r"Model\s+:\s+(.+)", cpu_items[-1])
        if match is not None:
            return match.group(1)

        for item in cpu_items:
            match = re.search(r"model name\s+:\s+(.+)", item)

            if match is not None:
                return match.group(1).strip()

    @staticmethod
    @callonce
    @exception_as_value(return_default=True)
    def __get_cpu_model_macos() -> Optional[str]:
        return (
            capped_check_output(["sysctl", "-n", "machdep.cpu.brand_string"])
            .decode("utf-8")
            .strip()
        )

    @staticmethod
    @callonce
    @exception_as_value(return_default=True)
    def __get_cpu_model_windows() -> Optional[str]:
        name = (
            capped_check_output(["wmic", "cpu", "get", "name"]).decode("utf-8").strip()
        )

        if name.startswith("Name"):
            name = name[4:].strip()

        return name

    @classmethod
    def machine(cls) -> Optional[str]:
        if platform.system() == "Linux":
            return cls.__get_cpu_model_linux()
        if platform.system() == "Darwin":
            return cls.__get_cpu_model_macos()
        if platform.system() == "Windows":
            return cls.__get_cpu_model_windows()

    @staticmethod
    @exception_as_value(return_default=True)
    def mac_address() -> Optional[str]:
        # Use netifaces
        return netifaces.ifaddresses(
            netifaces.gateways()["default"][netifaces.AF_INET][1]
        )[netifaces.AF_LINK][0]["addr"]

    @staticmethod
    @exception_as_value(return_default=True, default=False)
    def is_ethernet() -> bool:
        return netifaces.gateways()["default"][netifaces.AF_INET][1].startswith("eth")

    @staticmethod
    @callonce
    @exception_as_value(return_default=True)
    def __ssid_linux() -> Optional[str]:
        return capped_check_output(["iwgetid", "-r"]).decode("utf-8").strip()

    @staticmethod
    @callonce
    @exception_as_value(return_default=True)
    def __ssid_macos() -> Optional[str]:
        airport_output = map(
            functools.partial(str.split, sep=": "),
            map(
                str.strip,
                capped_check_output(
                    [
                        "/System/Library/PrivateFrameworks/Apple80211.framework/Versions/Current/Resources/airport",
                        "-I",
                    ]
                )
                .decode("utf-8")
                .strip()
                .split("\n"),
            ),
        )

        for field, value in airport_output:
            if field == "SSID":
                return value

    @staticmethod
    @callonce
    @exception_as_value(return_default=True)
    def __ssid_windows() -> Optional[str]:
        output = (
            capped_check_output(["netsh", "wlan", "show", "interfaces"])
            .decode("utf-8")
            .strip()
        )

        for line in output.split("\n"):
            line = line.strip()

            if line.startswith("SSID"):
                return line[4:].strip()[1:].strip()

        return None

    @classmethod
    def ssid(cls) -> Optional[str]:
        if platform.system() == "Linux":
            return cls.__ssid_linux()

        if platform.system() == "Darwin":
            return cls.__ssid_macos()

        if platform.system() == "Windows":
            return cls.__ssid_windows()

        return None

    @staticmethod
    def hostname() -> str:
        return socket.gethostname()

    @staticmethod
    @exception_as_value(return_default=True)
    def local_ip() -> Optional[str]:
        interface = netifaces.gateways()["default"][netifaces.AF_INET][1]

        return netifaces.ifaddresses(interface)[netifaces.AF_INET][0]["addr"]

    @staticmethod
    @callonce
    def core_count() -> Optional[int]:
        return os.cpu_count()

    @staticmethod
    @callonce
    def total_memory() -> int:
        return psutil.virtual_memory().total

    @classmethod
    def restart(cls):
        if platform.system() == "Linux":
            os.system("sudo reboot")
        elif platform.system() == "Darwin":
            os.system("reboot")
        elif platform.system() == "Windows":
            os.system("shutdown /r /t 1")

    @classmethod
    def shutdown(cls):
        if platform.system() == "Linux":
            os.system("sudo shutdown now")
        elif platform.system() == "Darwin":
            os.system("shutdown now")
        elif platform.system() == "Windows":
            os.system("shutdown /s /t 1")
