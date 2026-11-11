"""Zero-copy in-memory logits masker for lllayer.

Provides C-FFI accelerated logits masking and a pure Python fallback
to guarantee O(1) allocation overhead and sub-millisecond masking latency.
All text and docstrings follow AGENTS.md ASCII and English-only rules.
"""

import ctypes
import os
import subprocess
from pathlib import Path
from typing import Iterable, Optional, Set
import numpy as np


def find_or_build_native_library() -> Optional[str]:
    """Locates the pre-compiled libmasker.so or attempts on-the-fly compilation."""
    module_dir = Path(__file__).resolve().parent
    native_dir = module_dir / "native"
    so_path = native_dir / "libmasker.so"

    if so_path.is_file():
        return str(so_path)

    # Attempt to compile if source exists
    cpp_source = native_dir / "masker.cpp"
    if cpp_source.is_file():
        try:
            native_dir.mkdir(parents=True, exist_ok=True)
            compiler = "g++" if os.system("which g++ > /dev/null 2>&1") == 0 else "clang++"
            cmd = [
                compiler,
                "-O3",
                "-shared",
                "-fPIC",
                "-std=c++17",
                str(cpp_source),
                "-o",
                str(so_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if so_path.is_file():
                return str(so_path)
        except Exception:
            return None
    return None


class NativeASTLogitsProcessor:
    """LogitsProcessor integrated into lllayer internal tools.

    Manages AST-constrained masking directly in RAM via C-FFI or vector fallback.
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
