"""Benchmark ``SimpleUpdateModel.update_model``: the merge run on every report.

Run from the repo root (the oracle lives in ``tests/``)::

    .venv/bin/python scripts/bench_model_diff.py
    .venv/bin/python scripts/bench_model_diff.py --reports 50000 --rounds 7

A brand device model mirrors the printer's last report and folds each new,
partial report in. A Bambu ``push_status`` that only carries temperatures is
the common case: one or two of ~90 fields set, the rest ``None``, plus nested
default-factory sub-models (AMS, ...) that come along empty.

The bench folds a stream of such reports into a Bambu-shaped synthetic model
with the current ``update_model`` and with the frozen pre-optimisation walk
(``tests/_reference_update_model.py``), interleaved round by round on the same
data, and prints the best per-report time of each. When the SimplyPrint client
is importable it also runs its real ``BambuDeviceModel``.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import BaseModel, Field, create_model  # noqa: E402

from simplyprint_ws_client.common.model import (  # noqa: E402
    Atomic,
    ExtraInfo,
    SimpleUpdateModel,
)
from tests._reference_update_model import reference_update_model  # noqa: E402


class Tray(BaseModel, SimpleUpdateModel):
    id: Optional[str] = None
    tray_type: Optional[str] = None
    tray_color: Optional[str] = None
    nozzle_temp_min: Optional[int] = None
    nozzle_temp_max: Optional[int] = None
    remain: Optional[int] = None


class AmsUnit(BaseModel, SimpleUpdateModel):
    id: Optional[str] = None
    humidity: Optional[str] = None
    temp: Optional[str] = None
    tray: Optional[List[Tray]] = None


class Ams(BaseModel, SimpleUpdateModel):
    ams: Optional[List[AmsUnit]] = None
    ams_exist_bits: Optional[str] = None
    tray_now: Optional[str] = None
    tray_tar: Optional[str] = None
    version: Optional[int] = None


class Lights(BaseModel, SimpleUpdateModel):
    node: Optional[str] = None
    mode: Optional[str] = None


class Upgrade(BaseModel, SimpleUpdateModel):
    status: Optional[str] = None
    progress: Optional[str] = None
    message: Optional[str] = None


class Xcam(BaseModel, SimpleUpdateModel):
    allow_skip_parts: Optional[bool] = None
    first_layer_inspector: Optional[bool] = None
    print_halt: Optional[bool] = None


class PrintBase(BaseModel, SimpleUpdateModel):
    ams: Ams = Field(default_factory=Ams)
    upgrade_state: Upgrade = Field(default_factory=Upgrade)
    xcam: Xcam = Field(default_factory=Xcam)
    lights_report: Optional[List[Lights]] = None
    hms: Atomic[Optional[List[Dict[str, int]]]] = None
    nozzle_temper: Optional[float] = None
    bed_temper: Optional[float] = None
    chamber_temper: Optional[float] = None
    gcode_state: Optional[str] = None
    mc_percent: Optional[int] = None
    command: Optional[str] = None
    sequence_id: Optional[str] = None
    msg: Optional[int] = None
    t_utc: ExtraInfo[Optional[int]] = None


# Pad to the width of the real BambuPrintModel (~90 fields).
Print = create_model(
    "Print",
    __base__=PrintBase,
    **{f"field_{i}": (Optional[str], None) for i in range(76)},
)


class Info(BaseModel, SimpleUpdateModel):
    command: Optional[str] = None
    sequence_id: Optional[str] = None


class Device(BaseModel, SimpleUpdateModel):
    info: Optional[Info] = None
    print: Optional[Print] = None  # type: ignore[valid-type]
    system: Optional[Info] = None
    t_utc: ExtraInfo[Optional[int]] = None


def synthetic_case(n: int) -> Tuple[Any, List[Any]]:
    state = Device(
        print=Print(
            nozzle_temper=200.0,
            bed_temper=60.0,
            gcode_state="RUNNING",
            ams=Ams(ams=[AmsUnit(id="0", tray=[Tray(id=str(i)) for i in range(4)])]),
        )
    )
    reports = [
        Device(
            print=Print(
                nozzle_temper=200.0 + i % 7,
                bed_temper=60.0 + i % 3,
                command="push_status",
                sequence_id=str(i),
                t_utc=i,
            )
        )
        for i in range(n)
    ]
    return state, reports


def bambu_case(n: int) -> Optional[Tuple[Any, List[Any]]]:
    try:
        from simplyprint_client.printers.bambu.protocol.model.root import (
            BambuDeviceModel,
        )
    except Exception:  # absent, or needs a newer Python than this one
        return None
    import json

    state = BambuDeviceModel.model_validate_json(
        json.dumps({"print": {"nozzle_temper": 200.0, "bed_temper": 60.0}})
    )
    reports = [
        BambuDeviceModel.model_validate_json(
            json.dumps(
                {
                    "print": {
                        "nozzle_temper": 200.0 + i % 7,
                        "bed_temper": 60.0 + i % 3,
                        "command": "push_status",
                        "sequence_id": str(i),
                    }
                }
            )
        )
        for i in range(n)
    ]
    return state, reports


def time_merge(
    merge: Callable[[Any, Any], Any], state: Any, reports: List[Any]
) -> float:
    start = time.perf_counter()
    for report in reports:
        merge(state, report)
    return (time.perf_counter() - start) / len(reports) * 1e6


def bench(name: str, case: Tuple[Any, List[Any]], rounds: int) -> None:
    state, reports = case
    new_state, old_state = copy.deepcopy(state), copy.deepcopy(state)
    current = type(state).update_model
    new_times, old_times = [], []
    for _ in range(rounds):
        old_times.append(time_merge(reference_update_model, old_state, reports))
        new_times.append(time_merge(current, new_state, reports))
    old, new = min(old_times), min(new_times)
    print(
        f"{name:<28} before {old:7.2f} us/report   after {new:7.2f} us/report"
        f"   ({old / new:4.1f}x)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=int, default=20_000)
    parser.add_argument("--rounds", type=int, default=9)
    args = parser.parse_args()

    print(f"Python {sys.version.split()[0]}, {args.reports} reports x {args.rounds}")
    bench("synthetic bambu temp-only", synthetic_case(args.reports), args.rounds)
    real = bambu_case(args.reports)
    if real is not None:
        bench("BambuDeviceModel temp-only", real, args.rounds)


if __name__ == "__main__":
    main()
