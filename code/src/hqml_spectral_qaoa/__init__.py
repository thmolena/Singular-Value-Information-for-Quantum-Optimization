"""Deterministic structured-operator reproduction package."""

__all__ = ["reproduce"]


def reproduce() -> dict:
    """Run the registered experiment without importing the CLI eagerly."""

    from .experiment import reproduce as _reproduce

    return _reproduce()
