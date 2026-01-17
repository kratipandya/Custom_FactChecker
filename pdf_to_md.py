#!/usr/bin/env python3
"""
Convert a PDF to Markdown with page markers compatible with factchecker.py.

Output format per page:
*** Page {i}/{total} of "{title}" begins here ***
<page text>

Usage:
  python tools/pdf_to_md.py input.pdf [output.md]

Notes:
  - Tries PyPDF2 first (commonly available). Falls back to pypdf if needed.
  - Normalizes line endings to \n and strips excessive trailing whitespace.
  - Creates ./markdown if it doesn’t exist.
"""

import sys
import os
from pathlib import Path


def load_reader(pdf_path):
    # Try PyPDF2, then pypdf
    try:
        import PyPDF2  # type: ignore
        reader = PyPDF2.PdfReader(str(pdf_path))
        meta = reader.metadata or {}
        title = None
        # PyPDF2 metadata keys are like '/Title'
        if hasattr(meta, 'title') and meta.title:
            title = str(meta.title)
        elif isinstance(meta, dict):
            title = meta.get('/Title') or meta.get('Title')
        return reader, title, 'PyPDF2'
    except Exception:
        pass

    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(str(pdf_path))
        meta = reader.metadata or {}
        title = getattr(meta, 'title', None)
        if not title and isinstance(meta, dict):
            title = meta.get('/Title') or meta.get('Title')
        return reader, title, 'pypdf'
    except Exception as e:
        raise RuntimeError("Neither PyPDF2 nor pypdf is available to read PDFs.") from e


def extract_text_per_page(reader):
    pages_text = []
    total = len(reader.pages)
    for i in range(total):
        page = reader.pages[i]
        # Both PyPDF2 and pypdf expose extract_text()
        try:
            text = page.extract_text() or ''
        except Exception:
            text = ''
        # Normalize line endings and strip trailing spaces on each line
        normalized = "\n".join(line.rstrip() for line in (text or '').replace('\r\n', '\n').replace('\r', '\n').split('\n'))
        pages_text.append(normalized)
    return pages_text


def main(argv):
    if len(argv) < 2:
        print("Usage: python tools/pdf_to_md.py input.pdf [output.md]")
        return 2

    pdf_path = Path(argv[1])
    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return 1

    if len(argv) >= 3:
        out_path = Path(argv[2])
    else:
        out_dir = Path('./markdown')
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (pdf_path.stem + '.md')

    try:
        reader, title, backend = load_reader(pdf_path)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    if not title:
        title = pdf_path.stem

    pages_text = extract_text_per_page(reader)
    total = len(pages_text)

    parts = []
    for idx, text in enumerate(pages_text, start=1):
        header = f'*** Page {idx}/{total} of "{title}" begins here ***'
        parts.append(header)
        parts.append(text)

    content = "\n\n".join(parts) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(content, encoding='utf-8')

    print(f"Converted with {backend}: {pdf_path} -> {out_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main(sys.argv))

