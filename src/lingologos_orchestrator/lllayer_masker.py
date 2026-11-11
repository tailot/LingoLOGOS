"""Zero-copy in-memory logits masker for lllayer.

Provides C-FFI accelerated logits masking and a pure numpy fallback. Both branches
run on Linux and on Windows: the kernel is compiled only when a GCC compatible C++
toolchain is found, and the numpy fallback is the supported path when none is present.
The two branches implement the same masking relation, so a node decodes identically on
either host and only the per step latency differs.
All text and docstrings follow AGENTS.md ASCII and English-only rules.
"""

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set
import numpy as np

NATIVE_SOURCE_NAME = "masker.cpp"

# Environment switch used by deployments and by the parity tests to force the numpy
# branch even on a host where the C kernel can be built.
DISABLE_NATIVE_ENV_VAR = "LINGOLOGOS_DISABLE_NATIVE_KERNEL"


def native_disabled() -> bool:
    """Returns True when the numpy branch is forced through the environment switch."""
    return os.environ.get(DISABLE_NATIVE_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def library_filename(os_name: str, sys_platform: str) -> str:
    """Maps a host identification to the shared object name of the masking kernel.

    Windows uses a DLL, macOS a dylib, and every other POSIX host a shared object.
    """
    if os_name == "nt":
        return "masker.dll"
    if sys_platform == "darwin":
        return "libmasker.dylib"
    return "libmasker.so"


def native_library_path() -> Path:
    """Absolute path of the platform specific shared object inside the native folder."""
    return Path(__file__).resolve().parent / "native" / library_filename(os.name, sys.platform)


def compiler_candidates(os_name: str = os.name) -> List[str]:
    """Returns the GCC compatible C++ compilers found on this host, in preference order.

    The lookup goes through shutil.which, so no shell is spawned and no external probe
    utility is required. That keeps the probe valid in PowerShell and in CMD on Windows,
    where a POSIX shell probe always fails and writes noise onto the stderr stream that
    the orchestrator parses for a node.
    """
    if os_name == "nt":
        preferred = ("g++", "c++", "clang++", "x86_64-w64-mingw32-g++", "aarch64-w64-mingw32-g++")
    else:
        preferred = ("c++", "g++", "clang++")
    found: List[str] = []
    for name in preferred:
        resolved = shutil.which(name)
        if resolved and resolved not in found:
            found.append(resolved)
    return found


def build_command(compiler: str, source: Path, target: Path) -> List[str]:
    """Assembles the compiler invocation shared by every supported platform."""
    return [compiler, "-O3", "-shared", "-fPIC", "-std=c++17", str(source), "-o", str(target)]


def find_or_build_native_library(allow_build: bool = True) -> Optional[str]:
    """Locates the platform shared object, compiling it once when a toolchain is present.

    Returns None when the numpy branch is forced, when no compiler is installed, or when
    every candidate compiler fails. Compiler output is captured to DEVNULL rather than
    inherited, so a failed probe leaves nothing on the standard streams of a node whose
    stdout the orchestrator parses as JSON.
    """
    if native_disabled():
        return None

    target = native_library_path()
    if target.is_file():
        return str(target)

    source = target.parent / NATIVE_SOURCE_NAME
    if not allow_build or not source.is_file():
        return None

    for compiler in compiler_candidates():
        try:
            done = subprocess.run(
                build_command(compiler, source, target),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=180,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if done.returncode == 0 and target.is_file():
            return str(target)
    return None


class NativeASTLogitsProcessor:
    """LogitsProcessor integrated into lllayer internal tools.

    Manages AST-constrained masking directly in RAM through the C-FFI kernel when that
    kernel was located or built, and through a persistent numpy buffer otherwise. The
    selection is host independent and the masking result is identical on both branches.
    """

    def __init__(self, vocab_size: int, lib_path: Optional[str] = None):
        self.vocab_size = int(vocab_size)
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")

        self._lib = None
        self._state_ptr = None
        self._fallback_mask = None

        if lib_path is None:
            lib_path = find_or_build_native_library()

        if lib_path and Path(lib_path).is_file():
            try:
                self._lib = ctypes.CDLL(str(lib_path))
                self._lib.masker_init.restype = ctypes.c_void_p
                self._lib.masker_init.argtypes = [ctypes.c_uint32]

                self._lib.masker_update_allowed.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_uint32),
                    ctypes.c_uint32,
                ]

                self._lib.masker_apply_in_place.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                ]

                self._lib.masker_free.argtypes = [ctypes.c_void_p]

                self._state_ptr = self._lib.masker_init(self.vocab_size)
            except Exception:
                self._lib = None
                self._state_ptr = None

        if self._state_ptr is None:
            # Initialize fallback buffer in RAM
            self._fallback_mask = np.full(self.vocab_size, -np.inf, dtype=np.float32)

    @property
    def is_native(self) -> bool:
        """Returns True if the C-FFI accelerator kernel is active."""
        return self._state_ptr is not None

    def update_mask_from_ast_state(self, allowed_token_ids: Iterable[int]) -> None:
        """Updates admissible token indices in RAM without memory reallocation."""
        valid_ids = [idx for idx in allowed_token_ids if 0 <= idx < self.vocab_size]

        if self._state_ptr is not None:
            count = len(valid_ids)
            c_array = (ctypes.c_uint32 * count)(*valid_ids)
            self._lib.masker_update_allowed(self._state_ptr, c_array, count)
        else:
            self._fallback_mask.fill(-np.inf)
            if valid_ids:
                self._fallback_mask[valid_ids] = 0.0

    def __call__(self, input_ids: Iterable[int], logits: np.ndarray) -> np.ndarray:
        """Applies the volatile mask vector to the logits in-place."""
        assert logits.shape[-1] == self.vocab_size, (
            f"Logits trailing dimension {logits.shape[-1]} does not match vocab_size {self.vocab_size}"
        )

        if not logits.flags["C_CONTIGUOUS"] or logits.dtype != np.float32:
            logits = np.ascontiguousarray(logits, dtype=np.float32)

        if self._state_ptr is not None:
            c_logits_ptr = logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            self._lib.masker_apply_in_place(self._state_ptr, c_logits_ptr)
        else:
            logits += self._fallback_mask

        return logits

    def cleanup(self) -> None:
        """Frees contiguous C memory buffer immediately upon request completion."""
        if self._state_ptr is not None and self._lib is not None:
            self._lib.masker_free(self._state_ptr)
            self._state_ptr = None

    def __del__(self) -> None:
        self.cleanup()
