"""Matplotlib font helpers for demo scripts."""

from __future__ import annotations

from matplotlib import font_manager, rcParams


def configure_chinese_font() -> str | None:
    """Pick a commonly available CJK font to avoid Chinese glyph warnings."""
    candidates = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Songti SC",
        "STHeiti",
        "Heiti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Microsoft YaHei",
        "SimHei",
    ]

    installed = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in candidates:
        if font_name in installed:
            sans_serif = rcParams.get("font.sans-serif", [])
            rcParams["font.family"] = "sans-serif"
            rcParams["font.sans-serif"] = [font_name, *sans_serif]
            rcParams["axes.unicode_minus"] = False
            return font_name

    rcParams["axes.unicode_minus"] = False
    return None
