"""Actionable failures at the build, load and symbol boundaries."""

from pathlib import Path
import shlex
from typing import Optional, Sequence


class JitBuildError(RuntimeError):
    def __init__(self, detail: str, *, kernel_id: str, spec_fingerprint: str,
                 command: Sequence[str] = (), returncode: Optional[int] = None,
                 log_path: Optional[Path] = None) -> None:
        self.detail = detail
        self.kernel_id = kernel_id
        self.spec_fingerprint = spec_fingerprint
        self.command = tuple(command)
        self.returncode = returncode
        self.log_path = log_path
        super().__init__(detail)

    def __str__(self) -> str:
        return ("%s: %s; kernel=%s; spec=%s; command=%s; returncode=%s; log=%s"
                % (type(self).__name__, self.detail, self.kernel_id,
                   self.spec_fingerprint, shlex.join(self.command) or "<Python API>",
                   self.returncode, self.log_path or "<not started>"))


class JitCompilationError(JitBuildError):
    """Source validation or compiler invocation failed."""


class JitLoadError(JitBuildError):
    """Verified bytes could not be loaded into this process/device."""


class JitSymbolError(JitLoadError):
    """The requested entry point was not exported/resolved."""
