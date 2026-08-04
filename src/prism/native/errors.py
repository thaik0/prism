"""Stable public exceptions for Prism's native storage boundary."""

from __future__ import annotations


class NativeStoreError(Exception):
    """A structured failure reported by the native storage engine."""

    __slots__ = ("code", "message", "record_id", "offset", "path", "operation")

    def __init__(
        self,
        *,
        code: str,
        message: str,
        record_id: int | None,
        offset: int | None,
        path: str | None,
        operation: str | None,
    ) -> None:
        self.code = code
        self.message = message
        self.record_id = record_id
        self.offset = offset
        self.path = path
        self.operation = operation
        super().__init__(message)

    def __str__(self) -> str:
        operation = f" {self.operation}" if self.operation is not None else ""
        details = []
        if self.record_id is not None:
            details.append(f"record_id={self.record_id}")
        if self.offset is not None:
            details.append(f"offset={self.offset}")
        if self.path is not None:
            details.append(f"path={self.path}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"[{self.code}]{operation}: {self.message}{suffix}"
