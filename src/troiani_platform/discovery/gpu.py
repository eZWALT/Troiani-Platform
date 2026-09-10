from __future__ import annotations

import shutil
import subprocess
import xml.etree.ElementTree as ET
from typing import Iterable

from troiani_platform.discovery.nodes import current_node
from troiani_platform.discovery.processes import classify_process, command_for_pid, username_for_pid
from troiani_platform.models import GPUResource, Occupancy, ProcessInfo, utcnow


def infer_model(name: str) -> str:
    upper = name.upper()
    for token in ("H100", "H200", "B200", "A100", "A6000", "A40", "L40", "V100", "T4", "P100", "4090", "3090"):
        if token in upper:
            return token
    return name.split()[0] if name else "GPU"


def _text(node: ET.Element | None, default: str = "") -> str:
    if node is None or node.text is None:
        return default
    return node.text.strip()


def _number(text: str) -> float:
    cleaned = "".join(ch for ch in text if ch.isdigit() or ch in ".-")
    try:
        return float(cleaned) if cleaned else 0.0
    except ValueError:
        return 0.0


def _nvidia_smi_xml() -> str | None:
    binary = shutil.which("nvidia-smi")
    if not binary:
        return None
    try:
        return subprocess.check_output(
            [binary, "-q", "-x"],
            text=True,
            timeout=15,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.SubprocessError, OSError):
        return None


def _parse_processes(gpu_el: ET.Element, gpu_uuid: str, owned_users: Iterable[str]) -> tuple[ProcessInfo, ...]:
    procs: list[ProcessInfo] = []
    for proc in gpu_el.findall("./processes/process_info"):
        pid_text = _text(proc.find("pid"))
        if not pid_text.isdigit():
            continue
        pid = int(pid_text)
        used = _text(proc.find("used_memory"))
        name = _text(proc.find("process_name")) or "unknown"
        username = username_for_pid(pid)
        command = command_for_pid(pid)
        procs.append(
            ProcessInfo(
                pid=pid,
                name=name,
                gpu_uuid=gpu_uuid,
                memory_used_gb=_number(used) / 1024.0 if "MiB" in used or "MB" in used.upper() else _number(used),
                username=username,
                troiani_owned=classify_process(username, command, owned_users),
                command=command,
            )
        )
    return tuple(procs)


def _occupancy_from_processes(procs: tuple[ProcessInfo, ...], memory_used_gb: float, memory_busy_gb: float) -> Occupancy:
    if any(not p.troiani_owned for p in procs):
        return Occupancy.RESEARCHER
    if any(p.troiani_owned for p in procs):
        return Occupancy.TROIANI
    if memory_used_gb >= memory_busy_gb:
        return Occupancy.RESEARCHER
    return Occupancy.AVAILABLE


def discover_gpus(
    node: str | None = None,
    owned_users: Iterable[str] | None = None,
    memory_busy_gb: float = 2.0,
    xml: str | None = None,
) -> list[GPUResource]:
    payload = xml if xml is not None else _nvidia_smi_xml()
    if not payload:
        return []
    root = ET.fromstring(payload)
    node_name = current_node(node)
    owned = list(owned_users or ())
    gpus: list[GPUResource] = []
    for index, gpu_el in enumerate(root.findall("./gpu")):
        uuid = _text(gpu_el.find("uuid")) or f"gpu-{node_name}-{index}"
        name = _text(gpu_el.find("product_name"))
        mem_total = _text(gpu_el.find("./fb_memory_usage/total"))
        mem_used = _text(gpu_el.find("./fb_memory_usage/used"))
        sm_util = _text(gpu_el.find("./utilization/gpu_util"))
        mem_ctrl = _text(gpu_el.find("./utilization/memory_util"))
        temp = _text(gpu_el.find("./temperature/gpu_temp"))
        power = _text(gpu_el.find("./gpu_power_readings/power_draw")) or _text(
            gpu_el.find("./power_readings/power_draw")
        )
        memory_gb = _number(mem_total) / 1024.0 if "MiB" in mem_total else _number(mem_total)
        memory_used_gb = _number(mem_used) / 1024.0 if "MiB" in mem_used else _number(mem_used)
        procs = _parse_processes(gpu_el, uuid, owned)
        gpus.append(
            GPUResource(
                uuid=uuid,
                node=node_name,
                index=index,
                name=name,
                model=infer_model(name),
                memory_gb=round(memory_gb, 3),
                utilization=_number(sm_util),
                memory_used_gb=round(memory_used_gb, 3),
                memory_util=_number(mem_ctrl),
                temperature=_number(temp) if temp else None,
                power_w=_number(power) if power else None,
                compute_processes=procs,
                occupancy=_occupancy_from_processes(procs, memory_used_gb, memory_busy_gb),
                updated_at=utcnow(),
            )
        )
    return gpus
