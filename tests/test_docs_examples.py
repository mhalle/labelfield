"""Every ```python block in README.md and docs/*.md runs, and prints what its comments say.

Each block is executed on its own, in a fresh namespace, so every example must be
self-contained. A line of the form ``print(...)  # expected`` declares the expected output of
that print; when a block has such lines, its printed output must equal them, in order.
"""
import contextlib
import io
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FILES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]
BLOCK = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
EXPECT = re.compile(r"^\s*print\(.*\)\s+#\s?(.*)$")


def _blocks():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        for m in BLOCK.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            yield pytest.param(m.group(1), id=f"{path.relative_to(ROOT)}:{line}")


def test_docs_have_examples():
    assert len(list(_blocks())) >= 5


@pytest.mark.parametrize("code", list(_blocks()))
def test_example_runs(code):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(code, "<doc example>", "exec"), {"__name__": "__doc_example__"})
    expected = [m.group(1).rstrip() for m in map(EXPECT.match, code.splitlines()) if m]
    if expected:
        assert [ln.rstrip() for ln in buf.getvalue().splitlines()] == expected
