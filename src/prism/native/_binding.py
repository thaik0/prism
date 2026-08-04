"""Single audited import point for the private native extension."""

from __future__ import annotations

import importlib


NATIVE_INSTALL_ERROR = (
    "Prism's native extension is not installed. "
    "Install the project with native build support using `python3 -m pip install -e .`."
)


try:
    native = importlib.import_module("prism._native")
except ModuleNotFoundError as error:
    if error.name != "prism._native":
        raise
    raise ImportError(NATIVE_INSTALL_ERROR) from error
