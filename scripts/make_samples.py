"""Build degraded test PDFs from the clean public samples.

Real scanned PDFs are hard to source reliably, so we simulate them: render pages
to low-DPI grayscale bitmaps, add sensor noise, JPEG-compress, and store them as
image-only pages (no text layer). Because we start from a known-good PDF we also
have ground truth to compare OCR output against.

Outputs (in samples/):
  attention-scanned.pdf  - every page image-only (a "fully scanned" document)
  attention-mixed.pdf    - text-layer PDF where pages 3-4 are image-only
                           (common in practice: scanned appendix, signed page, ...)

Usage:  uv run python scripts/make_samples.py
"""

from pathlib import Path

import numpy as np
import pymupdf

SAMPLES = Path(__file__).resolve().parent.parent / "samples"
RNG = np.random.default_rng(42)


def degrade(page: pymupdf.Page, dpi: int = 110) -> bytes:
    """Render a page to a noisy, JPEG-compressed grayscale image."""
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    noisy = img.astype(np.int16) + RNG.normal(0, 18, img.shape).astype(np.int16)
    noisy = np.clip(noisy * 0.92 + 8, 0, 255).astype(np.uint8)  # slightly grey paper
    out = pymupdf.Pixmap(pymupdf.csGRAY, pix.width, pix.height, noisy.tobytes(), 0)
    return out.tobytes("jpeg", jpg_quality=55)


def rasterize(src: pymupdf.Document, pages_to_scan: set[int]) -> pymupdf.Document:
    out = pymupdf.open()
    for i, page in enumerate(src):
        if i in pages_to_scan:
            new = out.new_page(width=page.rect.width, height=page.rect.height)
            new.insert_image(new.rect, stream=degrade(page))
        else:
            out.insert_pdf(src, from_page=i, to_page=i)
    return out


def main() -> None:
    src = pymupdf.open(SAMPLES / "attention.pdf")
    rasterize(src, set(range(src.page_count))).save(SAMPLES / "attention-scanned.pdf", garbage=4, deflate=True)
    rasterize(src, {2, 3}).save(SAMPLES / "attention-mixed.pdf", garbage=4, deflate=True)
    print("wrote attention-scanned.pdf, attention-mixed.pdf")


if __name__ == "__main__":
    main()
