"""Control and inspect page cache for this experiment's segment files only."""

import ctypes
import mmap
import os
from pathlib import Path


def segments(data_root):
    root = Path(data_root).resolve()
    paths = sorted((root / "segments").glob("*.seg"))
    if not paths:
        raise ValueError(f"no experiment segment files under {root}")
    if any(path.is_symlink() for path in paths):
        raise ValueError("refusing symlinked segment files")
    return paths


def residency(data_root):
    total = resident = 0
    libc = ctypes.CDLL(None, use_errno=True)
    for path in segments(data_root):
        size = path.stat().st_size
        if not size:
            continue
        pages = (size + mmap.PAGESIZE - 1) // mmap.PAGESIZE
        with path.open("rb") as handle:
            mapping = mmap.mmap(handle.fileno(), size, access=mmap.ACCESS_COPY)
            try:
                pin = ctypes.c_char.from_buffer(mapping)
                address = ctypes.addressof(pin)
                del pin
                vector = (ctypes.c_ubyte * pages)()
                status = libc.mincore(ctypes.c_void_p(address), ctypes.c_size_t(size), vector)
                if status:
                    error = ctypes.get_errno()
                    raise OSError(error, os.strerror(error))
                total += pages
                resident += sum(value & 1 for value in vector)
            finally:
                mapping.close()
    return {"resident_pages": resident, "total_pages": total,
            "resident_fraction": resident / total if total else 0.0}


def prepare(data_root, state):
    if state == "warm":
        for path in segments(data_root):
            with path.open("rb") as handle:
                while handle.read(1024 * 1024):
                    pass
    elif state == "cold":
        for path in segments(data_root):
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
    else:
        raise ValueError("cache state must be warm or cold")
    result = residency(data_root)
    if state == "warm" and result["resident_fraction"] < 0.9:
        raise RuntimeError(f"could not warm experiment pages: {result}")
    if state == "cold" and result["resident_fraction"] > 0.1:
        raise RuntimeError(f"could not evict enough experiment pages: {result}")
    return result
