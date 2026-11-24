"""Which cores Holly runs on while he starts up.

A stopgap for an unstable CPU. On an i9-14900K, native crashes at start-up (access
violations in Python itself, at random places, mostly while the voice compiles and
warms up) came 7 times in 8 pinned to its performance cores and once in 8 on its
efficiency cores; the same crashes came from builds that had started cleanly days
before. With [cpu] startup = "efficiency", Holly loads and warms up on the efficiency
cores only, more slowly (the voice's load took 25 s, not 14), then runs on them all.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
from ctypes import wintypes


def _kernel32():
    """kernel32 with its handles typed: left as ints, a 64-bit handle is cut to 32 bits
    and the calls fail (Holly's own copy, so others' settings are left alone)."""
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    mask = ctypes.POINTER(ctypes.c_size_t)
    k32.GetProcessAffinityMask.argtypes = [wintypes.HANDLE, mask, mask]
    k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
    return k32


def efficiency_mask() -> int | None:
    """The efficiency cores' logical processors, as an affinity mask (Windows' CPU sets:
    those of the lowest efficiency class); None if every core is the same kind, or this
    isn't Windows."""
    if sys.platform != "win32":
        return None
    k32 = _kernel32()
    need = wintypes.ULONG(0)
    k32.GetSystemCpuSetInformation(None, 0, ctypes.byref(need), None, 0)
    buf = ctypes.create_string_buffer(need.value)
    if not k32.GetSystemCpuSetInformation(buf, need, ctypes.byref(need), None, 0):
        return None
    # SYSTEM_CPU_SET_INFORMATION: Size (DWORD), Type (DWORD), then the CPU set: Id (DWORD),
    # Group (WORD), LogicalProcessorIndex (BYTE), CoreIndex (BYTE), LastLevelCacheIndex
    # (BYTE), NumaNodeIndex (BYTE), EfficiencyClass (BYTE), ...
    by_class: dict[int, int] = {}
    raw, off = buf.raw, 0
    while off < need.value:
        size = int.from_bytes(raw[off:off + 4], "little")
        logical, efficiency = raw[off + 14], raw[off + 18]
        by_class[efficiency] = by_class.get(efficiency, 0) | (1 << logical)
        off += size
    return by_class[min(by_class)] if len(by_class) > 1 else None


@contextlib.contextmanager
def starting_on(cores: str):
    """While in this block, the whole process runs on `cores` ("efficiency", or "all" to
    leave it be); afterwards, wherever it could before."""
    mask = efficiency_mask() if cores == "efficiency" else None
    if not mask:
        yield
        return
    k32 = _kernel32()
    process = k32.GetCurrentProcess()
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    k32.GetProcessAffinityMask(process, ctypes.byref(allowed), ctypes.byref(system))
    k32.SetProcessAffinityMask(process, mask & allowed.value)
    try:
        yield
    finally:
        k32.SetProcessAffinityMask(process, allowed.value)


def free(child) -> None:
    """Let a child process (a subprocess.Popen) run on every core: started while Holly
    starts up on the efficiency cores, it would stay on them for good (it inherits them)."""
    if sys.platform != "win32" or not getattr(child, "_handle", None):
        return
    k32 = _kernel32()
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system))
    k32.SetProcessAffinityMask(int(child._handle), system.value)
