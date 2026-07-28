"""OCR must be an enhancement, never a reason to lose a render.

Three distinct ways OCR can be unavailable, all of which have to degrade to saliency-only:
the Python binding is missing, the tesseract binary is missing, or tesseract is installed
but broken. The third is the nasty one — a misconfigured tesseract exits nonzero and
pytesseract then raises UnicodeDecodeError while decoding tesseract's own stderr, which is
not an error type anyone would think to catch.
"""
import sys
import types

import numpy as np
import pytest

from reel_maker.reframe import OcrUnavailable, text_mask

FRAME = np.zeros((80, 120, 3), np.uint8)


def test_missing_pytesseract_raises_ocr_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pytesseract", None)  # forces ImportError on import
    with pytest.raises(OcrUnavailable) as e:
        text_mask(FRAME, 45)
    assert "pip install" in str(e.value)


def _fake_pytesseract(exc: Exception) -> types.ModuleType:
    mod = types.ModuleType("pytesseract")

    class TesseractNotFoundError(Exception):
        pass

    class Output:
        DICT = "dict"

    def image_to_data(*_a, **_k):
        raise exc

    mod.TesseractNotFoundError = TesseractNotFoundError
    mod.Output = Output
    mod.image_to_data = image_to_data
    return mod


def test_binary_not_found_raises_ocr_unavailable(monkeypatch):
    mod = _fake_pytesseract(RuntimeError("placeholder"))
    mod.image_to_data = lambda *a, **k: (_ for _ in ()).throw(mod.TesseractNotFoundError())
    monkeypatch.setitem(sys.modules, "pytesseract", mod)
    with pytest.raises(OcrUnavailable) as e:
        text_mask(FRAME, 45)
    assert "not on PATH" in str(e.value)


@pytest.mark.parametrize("exc", [
    UnicodeDecodeError("utf-8", b"\x89", 0, 1, "invalid start byte"),
    RuntimeError("tesseract exited with status 1"),
])
def test_broken_tesseract_degrades_instead_of_exploding(monkeypatch, exc):
    """The regression this file exists for: a broken-but-present tesseract used to
    propagate a raw UnicodeDecodeError and abort the whole render."""
    monkeypatch.setitem(sys.modules, "pytesseract", _fake_pytesseract(exc))
    with pytest.raises(OcrUnavailable) as e:
        text_mask(FRAME, 45)
    assert "failed to run" in str(e.value)
    assert "TESSDATA_PREFIX" in str(e.value)
