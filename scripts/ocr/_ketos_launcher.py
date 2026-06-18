"""Wrapper that invokes ``ketos`` with a few monkey-patches in place.

Three upstream behaviours are patched here:

1) ``rich.Console.clear_live``: kraken's ``RichProgressBar`` callback
   calls ``Console.clear_live()`` on ``on_sanity_check_start`` before any
   Live display has been pushed onto the console's ``_live_stack``. In
   recent ``rich`` releases ``clear_live`` unconditionally pops from that
   stack, so the very first sanity check crashes with
   ``IndexError: pop from empty list``. We replace it with a no-op when
   the stack is empty.

2) ``kraken.lib.lineest.CenterNormalizer.dewarp``: builds a list of
   per-column slices and calls ``np.array(list, dtype=...).T``. When the
   line-estimator's centre clips near the top/bottom edge, some columns
   come back shorter than ``2*self.r``. NumPy < 2.0 silently coerced the
   inhomogeneous list to an object array; NumPy >= 2.0 raises
   ``ValueError: setting an array element with a sequence``. Kraken 6.0.2
   pins ``numpy>=2.0`` so downgrading isn't an option. We pad short
   columns to ``2*self.r`` with ``cval`` so the stack succeeds — same
   end behaviour as the legacy object-array path.

3) ``RecognitionModel.configure_callbacks`` (kraken's ``EarlyStopping``):
   kraken creates the callback without passing ``min_delta``, so Lightning
   defaults to ``min_delta=0.0`` — any positive change in
   ``val_accuracy`` counts as an improvement and resets the patience
   counter. At high val_accuracy (e.g. 0.99785 → 0.99786 — one extra
   correct character out of ~100k) that turns into a near-infinite loop
   of noise-driven "improvements". We replace it with a callback that
   uses ``EARLY_STOP_MIN_DELTA`` (default 0.0005 ≈ 0.05% absolute
   improvement) as the threshold, override via the
   ``KETOS_EARLY_STOP_MIN_DELTA`` environment variable.

All patches are applied before importing ``kraken.ketos`` so the
patched methods are in place when Lightning / Kraken instantiate their
machinery.

Usage (drop-in for the ``ketos`` entry point):

    python scripts/ocr/_ketos_launcher.py -d cpu train -f path ...
"""

import os
import sys

import numpy as np
from kraken.lib import lineest
from kraken.lib import train as kraken_train
from lightning.pytorch.callbacks import EarlyStopping
from rich.console import Console

# Minimum val_accuracy improvement (absolute) required for early stopping
# to consider an epoch a "real" improvement. 0.0005 = 0.05 percentage
# points, which is well above the per-epoch noise floor at the
# late-convergence regime (val_accuracy > 99%). Override via env var:
#
#     KETOS_EARLY_STOP_MIN_DELTA=0.001 make finetune_ocr ...
EARLY_STOP_MIN_DELTA = float(os.environ.get("KETOS_EARLY_STOP_MIN_DELTA", "0.0005"))


def _safe_clear_live(self):
    with self._lock:
        if self._live_stack:
            self._live_stack.pop()


def _safe_dewarp(self, img, cval=0, dtype=np.dtype("f")):
    if img.shape != self.shape:
        raise Exception("Measured and dewarp image shapes different")
    h, w = img.shape
    padded = np.vstack([cval * np.ones((h, w)), img, cval * np.ones((h, w))])
    center = self.center + h
    expected_len = 2 * self.r
    cols = []
    for i in range(w):
        col = padded[center[i] - self.r : center[i] + self.r, i]
        if col.shape[0] != expected_len:
            # Edge-clipped column: pad with the background value so the
            # stack is rectangular. Matches the pre-numpy-2.0 silent
            # object-array fallback that kraken originally relied on.
            pad = np.full(expected_len - col.shape[0], cval, dtype=col.dtype)
            col = np.concatenate([col, pad])
        cols.append(col)
    return np.array(cols, dtype=dtype).T


def _configure_callbacks_with_min_delta(self):
    """Replacement for ``RecognitionModel.configure_callbacks``.

    Same shape as the original (only a single ``EarlyStopping`` callback
    in quit='early' mode) but passes ``min_delta=EARLY_STOP_MIN_DELTA``
    so val_accuracy jitter at the noise floor doesn't keep training
    running indefinitely.
    """
    callbacks = []
    if self.hyper_params["quit"] == "early":
        callbacks.append(
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=self.hyper_params["lag"],
                stopping_threshold=1.0,
                min_delta=EARLY_STOP_MIN_DELTA,
            )
        )
    return callbacks


# Apply the patches unconditionally so spawned DataLoader workers, which
# re-import this script under __mp_main__, also benefit.
Console.clear_live = _safe_clear_live
lineest.CenterNormalizer.dewarp = _safe_dewarp
kraken_train.RecognitionModel.configure_callbacks = _configure_callbacks_with_min_delta


if __name__ == "__main__":
    # Guard the CLI call: on macOS PyTorch DataLoader uses `spawn` to
    # start worker processes, which re-imports this file under
    # __mp_main__. Without the guard, every worker would call cli() and
    # try to run ketos training again, recursively.
    sys.argv[0] = "ketos"
    from kraken.ketos import cli

    cli()
