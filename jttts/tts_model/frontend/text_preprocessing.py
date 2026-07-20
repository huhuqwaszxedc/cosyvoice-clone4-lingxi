#!/usr/bin/python
# -*- encoding: utf-8 -*-

"""Lightweight text cleanup shared by the TTS front end."""

import re


_SUPPORTED_HTML_ENTITIES = {
    # Quotation marks are punctuation and do not need to be spoken.
    "ldquo": "",
    "rdquo": "",
    "lsquo": "",
    "rsquo": "",
    "quot": "",
    "apos": "",
    # Comparison and arithmetic entities are converted to symbols first,
    # then verbalized together with literal symbols below.
    "lt": "<",
    "gt": ">",
    "le": "≤",
    "ge": "≥",
    "ne": "≠",
    "times": "×",
    "divide": "÷",
    "plusmn": "±",
}

_SUPPORTED_HTML_ENTITY = re.compile(
    r"&("
    + "|".join(
        sorted(_SUPPORTED_HTML_ENTITIES, key=len, reverse=True)
    )
    + r")(?:;|(?![A-Za-z0-9]))"
)

_BRACKET_MARKS = str.maketrans("", "", "()[]{}（）【】｛｝")


def normalize_html_entities(text: str) -> str:
    """Normalize only explicitly supported entities, with or without ``;``.

    Layout and unit entities such as ``&nbsp;``, ``&hellip;`` and ``&deg;``
    intentionally remain unchanged.
    """

    return _SUPPORTED_HTML_ENTITY.sub(
        lambda match: _SUPPORTED_HTML_ENTITIES[match.group(1)],
        text,
    )


def verbalize_math_symbols(text: str) -> str:
    """Convert comparison and arithmetic symbols to readable Chinese text."""

    replacements = (
        ("<=", "小于等于"),
        (">=", "大于等于"),
        ("＜＝", "小于等于"),
        ("＞＝", "大于等于"),
        ("≤", "小于等于"),
        ("≥", "大于等于"),
        ("≦", "小于等于"),
        ("≧", "大于等于"),
        ("!=", "不等于"),
        ("≠", "不等于"),
        ("<", "小于"),
        (">", "大于"),
        ("＜", "小于"),
        ("＞", "大于"),
        ("＝", "等于"),
        ("×", "乘"),
        ("÷", "除"),
        ("±", "正负"),
    )
    for symbol, spoken_text in replacements:
        text = text.replace(symbol, spoken_text)
    return text


def strip_bracket_marks(text: str) -> str:
    """Remove bracket marks while retaining their spoken content."""

    return text.translate(_BRACKET_MARKS)
