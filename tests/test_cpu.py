"""The start-up cores stopgap (app/cpu.py)."""

import ctypes
import sys

import pytest

from app import cpu


def mask() -> int:
    k32 = cpu._kernel32()
    allowed, system = ctypes.c_size_t(), ctypes.c_size_t()
    k32.GetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.byref(allowed), ctypes.byref(system))
    return allowed.value


def test_all_cores_leaves_it_be():
    with cpu.starting_on("all"):
        pass  # nothing to check off Windows; on it, nothing changes


@pytest.mark.skipif(sys.platform != "win32" or not cpu.efficiency_mask(), reason="needs a CPU with two kinds of core")
def test_efficiency_cores_while_starting_then_all():
    before = mask()
    with cpu.starting_on("efficiency"):
        assert mask() == cpu.efficiency_mask() & before
    assert mask() == before
