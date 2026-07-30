"""Line-image preprocessing: aspect-ratio-preserving **pad** (default) vs **stretch**.

TrOCR's ``ViTImageProcessor`` resizes every image to a fixed square
(384x384 for ``microsoft/trocr-base-handwritten``, 224x224 for Swin) by a
**non-uniform stretch** — it destroys the line's aspect ratio (a ~400x39
line strip is squeezed horizontally ~0.9x and stretched vertically ~12x, so
glyphs become tall and thin). The stretch is applied identically at train and
inference, so a model *can* learn it, but it (a) distorts glyph proportions and
(b) applies a *different* stretch factor to synthetic renders (~1000x115) than
to real crops (~400x39).

``pad`` mode instead scales the line **preserving aspect ratio** to fit the
processor's target size, then centre-pads the remainder with ``fill`` — so
glyph proportions are kept and synthetic/real lines land in the encoder's view
at the same shape regardless of their native pixel size. Because the padded
image already equals the processor's target size, the processor's own resize
becomes a no-op.

Kept behind a ``mode`` flag so **stretch-vs-pad can be ablated** (spec §6.5.18).
Default is ``pad``.
"""

from __future__ import annotations

from PIL import Image

RESIZE_MODES = ("pad", "stretch")
DEFAULT_RESIZE_MODE = "pad"
# White background — matches the light parchment/blank margin of the line crops.
DEFAULT_PAD_FILL = (255, 255, 255)


def target_hw(image_processor) -> tuple[int, int]:
    """Read the processor's target (height, width) from its ``size`` config."""
    s = image_processor.size

    def _get(key):
        if isinstance(s, dict):
            return s.get(key)
        return getattr(s, key, None)

    h, w = _get("height"), _get("width")
    if h is None or w is None:
        edge = _get("shortest_edge") or _get("longest_edge")
        h = w = edge
    return int(h), int(w)


def prepare_image(
    image: Image.Image,
    image_processor,
    mode: str = DEFAULT_RESIZE_MODE,
    fill: tuple[int, int, int] = DEFAULT_PAD_FILL,
) -> Image.Image:
    """Return a PIL image ready to hand to ``image_processor``.

    - ``stretch``: return the image unchanged; the processor does its default
      square stretch (aspect ratio destroyed).
    - ``pad``: scale preserving aspect ratio to fit the processor's target size
      and centre-pad with ``fill`` (aspect ratio preserved). The processor's
      resize then no-ops on the already-target-sized canvas.
    """
    if mode == "stretch":
        return image
    if mode != "pad":
        raise ValueError(f"resize mode must be one of {RESIZE_MODES}, got {mode!r}")

    target_h, target_w = target_hw(image_processor)
    ow, oh = image.size
    scale = min(target_w / ow, target_h / oh)
    new_w, new_h = max(1, round(ow * scale)), max(1, round(oh * scale))
    resized = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), fill)
    canvas.paste(resized, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return canvas
