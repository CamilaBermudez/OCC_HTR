"""Wrapper that invokes ``ketos`` after patching ``rich.Console.clear_live``.

Kraken's ``RichProgressBar`` callback calls ``Console.clear_live()`` on
``on_sanity_check_start`` before any Live display has been pushed onto
the console's ``_live_stack``. In recent ``rich`` releases ``clear_live``
unconditionally pops from that stack, so the very first sanity check
crashes with ``IndexError: pop from empty list``.

The fix here is a one-line monkey-patch that turns ``clear_live`` into a
no-op when the stack is empty. We apply it before importing
``kraken.ketos`` so the patched method is in place by the time Lightning
instantiates the progress bar.

Usage (drop-in for the ``ketos`` entry point):

    python scripts/ocr/_ketos_launcher.py -d cpu train -f path ...
"""

import sys

from rich.console import Console


def _safe_clear_live(self):
    with self._lock:
        if self._live_stack:
            self._live_stack.pop()


# Apply the patch unconditionally so spawned DataLoader workers, which
# re-import this script under __mp_main__, also benefit.
Console.clear_live = _safe_clear_live


if __name__ == "__main__":
    # Guard the CLI call: on macOS PyTorch DataLoader uses `spawn` to
    # start worker processes, which re-imports this file under
    # __mp_main__. Without the guard, every worker would call cli() and
    # try to run ketos training again, recursively.
    sys.argv[0] = "ketos"
    from kraken.ketos import cli

    cli()
