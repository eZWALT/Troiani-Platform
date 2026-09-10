from __future__ import annotations


class PlatformError(Exception):
    """Base error for the control plane and workers."""


class JobSpecError(PlatformError, ValueError):
    pass


class TransitionError(PlatformError, ValueError):
    pass


class NotFound(PlatformError, KeyError):
    pass


class CheckpointError(PlatformError):
    pass


class AuthError(PlatformError):
    pass


class PolicyError(PlatformError):
    pass
