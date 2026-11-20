"""Amplitude storage types shared by portable and runtime state boundaries."""
from __future__ import annotations

import numpy as np

SUPPORTED_DTYPES = frozenset({"float32", "float64", "complex64", "complex128"})


def amplitude_array(values, dtype=None):
    """Retain floating storage precision; integer/bool inputs use float64.

    Explicit real storage must never silently discard an imaginary component.
    Cast overflow is rejected at the boundary, rather than reaching a backend.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError("state amplitudes must be one-dimensional")
    if array.dtype.kind not in "buifc":
        raise ValueError("state amplitudes must be numeric")
    if dtype is None:
        dtype = "float64" if array.dtype.kind in "bui" else array.dtype.name
    if not isinstance(dtype, str) or dtype not in SUPPORTED_DTYPES:
        raise ValueError("state dtype must be float32, float64, complex64, or complex128")
    if dtype.startswith("float") and array.dtype.kind == "c":
        if np.any(array.imag != 0):
            raise ValueError("real state dtype cannot contain imaginary amplitudes")
        array = array.real
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(dtype, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError("state amplitudes must be finite in the requested dtype")
    return array
