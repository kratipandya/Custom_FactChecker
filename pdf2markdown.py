#!/usr/bin/env python3
"""
PDF to Markdown converter using OpenRouter/Gemini vision models.

Refactor highlights:
- Cache is a single SQLite table with exactly three columns:
  filename TEXT, page_number INTEGER, content TEXT
  PRIMARY KEY (filename, page_number)
- Progress bar starts at the number of cached pages immediately (e.g., 50/80).
- Default concurrency is 3; configurable with --parallel.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import sqlite3
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait

import fitz  # type: ignore
import requests
from requests import exceptions as requests_exceptions

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore

try:
    from PIL import Image, ImageChops, ImageOps  # type: ignore
except Exception:
    Image = None  # type: ignore
    ImageChops = None  # type: ignore
    ImageOps = None  # type: ignore


# --------------------
# Config & Defaults
# --------------------

BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_FALLBACK_KEY = "AIzaSyDAqZkH8NRLQcuggwTrX4pEg-0EAyUhKo8"
GEMINI_THINKING_BUDGET = 24_576

DEFAULT_PROMPT = """
Your Task: Lossless OCR → Markdown with PRINTED VALUES + DIGITIZED ESTIMATES

OBJECTIVE
Convert the provided page image(s) into ONE Markdown document that preserves all meaningful information with MINIMAL or NO LOSS, suitable for scientific work.
• Do not insert meta-information not specified here, like your personal annotations or comments to your transcription.
• Transcribe — do NOT summarize, paraphrase, or interpret trends/causes.
• Capture EVERYTHING that conveys meaning (text, numbers, figures).
• Ignore decorative elements (rules, background shading, page frames, drop caps, ornaments, unreferenced page numbers).
• Keep the source language, punctuation, and locale number formatting (e.g., German thousands “.”, decimals “,”).

GLOBAL RULES
1) Reading order: title → headings → body → figures/plots → tables → footnotes. If ambiguous, add a one-line NOTE describing the two plausible orders.
2) Headings: map visual hierarchy to Markdown #…###### and preserve printed numbering (e.g., “1.4.3”).
3) Paragraphs & lists: keep line breaks and list markers.
   • Hyphenation at line ends: if a word is split across lines and clearly continues, JOIN it (e.g., “Langzeit- arbeitslosigkeit” → “Langzeitarbeitslosigkeit”). Keep true mid-line hyphens.
4) Numbers & units: keep exactly as printed (units, symbols, spacing, separators).
5) Footnotes: use Markdown footnotes [^n] where marks appear; define them verbatim at the end.
6) Links/URLs: transcribe fully; if a character is unclear, replace with □ and add a brief note.
7) Text inside images (captions, axis labels, legends, annotations, “Quelle/Source”, “Hinweis/Note”) must be transcribed verbatim in that figure’s block.
8) Non-text objects (each figure/plot/map/photo/icon-with-meaning) must have TWO representations in this order:
   A. a machine-readable TABLE block; then
   B. a DETAILED non-inferential description (enumerate visible elements; no trend/causal statements).
9) Uncertain tokens in TEXT: mark as ⟦uncertain:"tok?en"⟧ + brief reason in a note.
10) Numeric consistency across the page:
    • When the same concept appears in multiple places (body vs. figure/table), transcribe all occurrences.
    • If values disagree, DO NOT resolve; add:
      > Inconsistency note: "<field>" in body = <value>; in figure/table = <value>.

PLOT DIGITIZATION (CRITICAL) — PRINTED values + DIGITIZED estimates

Goal: capture BOTH (1) precise numbers that are PRINTED inside/around the figure, AND (2) geometry-based DIGITIZED estimates at every printed x tick for every series/segment. Never invent unsupported data.

A) Scope to digitize
• Chart types: line, bar (clustered/stacked), area, scatter/dot, histogram, box/violin, heatmap, dot/interval, small multiples/panels.
• For each printed x tick/category and for EACH series/segment: estimate the y value (height/position).
• For shaded bands/intervals: also estimate LOWER and UPPER at the same x ticks if visually readable.

B) Calibration (axis↔pixel) — MANDATORY & REPRODUCIBLE
• Calibrate on the INNER plotting box (exclude margins).
• Record ≥3 labeled ticks per axis when available and include pixel anchors.
• Assume linear scales unless a non-linear scale is explicitly printed (e.g., “log₁₀”); state the scale used.
• Use printed tick labels as anchors; read intermediate positions by linear interpolation.

C) Units printed ABOVE/AROUND axes
• Treat figure-level unit lines (e.g., “in Tausend”) as the y-axis unit even if the axis itself has no title.
• Explicitly record the unit source.

D) Precision & uncertainty
• Uncertainty (±) = max(½ × smallest y-axis major tick step, 1% of full y-range), expressed in figure units.
• Round to the same precision as axis tick labels (whole, one decimal, thousands, etc.).
• If occluded/indistinct: use “n. a. (occluded)” and explain.

E) Printed vs. Digitized — ALWAYS output BOTH
• Keep two separate numeric sections per figure:
  1) Printed numeric labels (verbatim): every number printed in or tied to the figure (data labels, annotation numbers, inset table values, labeled endpoints). If none: “none printed.”
  2) Digitized estimates (geometry-based): values at EVERY printed x tick for EVERY series/segment.
• If a printed value exists for (series, x): also include it in the digitized table with Uncertainty = 0 and Method note = “printed value copied”.

F) Sanity checks (auto-note if triggered)
• Ensure estimated values lie within axis range.
• If a series visually touches a major gridline but the estimate deviates by more than one uncertainty band, add:
  > Calibration check: series appears near <tick>; estimate <value>. Re-verify anchors.

OUTPUT FORMAT (USE EXACTLY THIS STRUCTURE)

Keep sections even if empty (use “none” or “n. a.” explicitly).

# <Top-level heading as printed>

<Body text paragraphs, verbatim>

## Figures

**Figure <number>: <caption/title as printed>**
*Table representation (specification):*
| Element | Value |
|---|---|
| Plot type | <e.g., multi-series line / bar / scatter / …> |
| Panels | <none / A–B–C with panel titles> |
| Plot box (px) | left=<L>, right=<R>, top=<T>, bottom=<B> |
| Axes | <primary/secondary; orientation> |
| Axis – x title | <text or “none printed”> |
| Axis – x ticks (as printed) | <comma-separated> |
| Axis – y title | <text or “none printed”> |
| Axis – y ticks (as printed) | <comma-separated> |
| Figure unit line | <text or “none”> |
| Axis – y unit source | <"axis title" | "figure unit line" | "both"> |
| Secondary axis | <title & ticks or “none”> |
| Legend entries (as printed) | <ordered list> |
| Series count | <integer> |
| Bands/intervals | <as printed or “none”> |
| Reference lines/targets | <as printed or “none”> |
| Source inside figure | <text> |
| Notes inside figure | <text or “none”> |

*Series overview (legend-mapped):*
| Series id | Printed name | Style (as labeled) | Notes |
|---|---|---|---|
| 1 | <text> | <solid/dashed; marker shape if printed> | <e.g., “saisonbereinigt” if printed> |
| 2 | … | … | … |

*Printed numeric labels (verbatim; write “none printed” if absent):*
| Series id | x label (as printed) | y (printed) | Unit (as printed) | Where printed | Note |
|---|---|---:|---|---|---|
| <id> | <Jan/2019/…> | <427,1> | <ppm/Tsd./%> | data label / annotation / inset table | <…> |
| … | … | … | … | … | … |

*Digitized estimates (geometry-based; include ALL printed x ticks for ALL series):*
| Series id | x label (as printed) | y (estimated) | Unit (as printed) | Uncertainty (±) | Method note |
|---|---|---:|---|---:|---|
| <id> | <2019> | <2.650> | Tsd. | 50 | intersection with line |
| <id> | <2020> | <…> | <…> | <…> | <…> |
| … | … | … | … | … | … |

*Digitization quality (required):*
- Calibration anchors:
  • y: <tick1 label>↦px(<y1>), <tick2 label>↦px(<y2>), <tick3 label>↦px(<y3>) …
  • x: <tick1 label>↦px(<x1>), <tick2 label>↦px(<x2>), …  
- Mapping/scale: <linear/log…>. Fit RMSE (px): <…>.
- Resolution: <px per major y tick>.
- Confidence: <high/medium/low>. Assumptions: <…>.

*Detailed non-inferential description:*
- <Enumerate literal visual elements: number of lines/bars/points; presence and position of labels; axis ranges exactly as printed; callouts; shading; arrows. No interpretations.>

**Figure <number>: <next figure>**
<repeat the same block>

## Tables (true page tables)
• Rebuild printed tables as GitHub-flavored Markdown. If a table has multi-row headers, reproduce them exactly as printed using multiple header rows.

**Table <number>: <caption as printed>**
|                     | <Header col 2 line 1> | <Header col 3 line 1> | <Header col 4 group> | <Header col 5 group> |
|                     | <Header col 2 line 2> | <Header col 3 line 2> | <Header 4 line 2>    | <Header 5 line 2>    |
|---|---:|---:|---:|---:|
| <row label>         | <value> | <value> | <value> | <value> |
| Quelle/Source: <as printed> |

## Footnotes
[^1]: <verbatim footnote text>
[… additional footnotes …]

EXAMPLE — ONE FIGURE WITH BOTH PRINTED VALUES & DIGITIZED ESTIMATES (fabricated; format only)

**Figure 2: Monatswerte der Arbeitslosenquote 2024 (in %)**
*Table representation (specification):*
| Element | Value |
|---|---|
| Plot type | Single-series line chart |
| Panels | none |
| Plot box (px) | left=84, right=640, top=140, bottom=720 |
| Axes | primary only |
| Axis – x title | Monat |
| Axis – x ticks (as printed) | Jan, Feb, Mär, Apr, Mai, Jun, Jul, Aug, Sep, Okt, Nov, Dez |
| Axis – y title | Arbeitslosenquote (in %) |
| Axis – y ticks (as printed) | 4,0; 4,5; 5,0; 5,5 |
| Figure unit line | none |
| Axis – y unit source | axis title |
| Secondary axis | none |
| Legend entries (as printed) | Arbeitslosenquote |
| Series count | 1 |
| Bands/intervals | none |
| Reference lines/targets | none |
| Source inside figure | Quelle: Musteramt |
| Notes inside figure | Werte gerundet auf 0,1 %-Punkte (gedruckt) |

*Series overview (legend-mapped):*
| Series id | Printed name | Style (as labeled) | Notes |
|---|---|---|---|
| 1 | Arbeitslosenquote | durchgezogene Linie mit Kreis-Markern | — |

*Printed numeric labels (verbatim):*
| Series id | x label (as printed) | y (printed) | Unit (as printed) | Where printed | Note |
|---|---|---:|---|---|---|
| 1 | Jan | 4,3 | % | data label | — |
| 1 | Jun | 4,9 | % | data label | — |
| 1 | Dez | 4,6 | % | data label | — |

*Digitized estimates (geometry-based; ALL months):*
| Series id | x label (as printed) | y (estimated) | Unit (as printed) | Uncertainty (±) | Method note |
|---|---|---:|---|---:|---|
| 1 | Jan | 4,3 | % | 0,0 | printed value copied |
| 1 | Feb | 4,4 | % | 0,05 | intersection with line |
| 1 | Mär | 4,6 | % | 0,05 | intersection with line |
| 1 | Apr | 4,7 | % | 0,05 | intersection with line |
| 1 | Mai | 4,8 | % | 0,05 | intersection with line |
| 1 | Jun | 4,9 | % | 0,0 | printed value copied |
| 1 | Jul | 5,0 | % | 0,05 | intersection with line |
| 1 | Aug | 5,1 | % | 0,05 | intersection with line |
| 1 | Sep | 5,0 | % | 0,05 | intersection with line |
| 1 | Okt | 4,9 | % | 0,05 | intersection with line |
| 1 | Nov | 4,7 | % | 0,05 | intersection with line |
| 1 | Dez | 4,6 | % | 0,0 | printed value copied |

*Digitization quality (required):*
- Calibration anchors:
  • y: 4,0↦px(720), 5,5↦px(140)  
  • x: Jan↦px(84), Dez↦px(640)
- Mapping/scale: linear. Fit RMSE (px): 0 (two-point).
- Resolution: ≈40 px per 0,1 %-Punkt.
- Confidence: high. Assumptions: marker centers used.

*Detailed non-inferential description:*
- Eine durchgezogene Linie mit Kreis-Markern verbindet die Monate Jan–Dez.
- Gedruckte numerische Labels erscheinen bei Jan, Jun, Dez; übrige Monate ohne Zahlen.
- Legende “Arbeitslosenquote” oben rechts; keine Bänder oder Referenzlinien.

EXAMPLE — MULTI-ROW TABLE HEADER (format only)

**Table A: Langzeitarbeitslosigkeit**
|                     | August 2025 | Anteil an allen Arbeitslosen | Veränderung Vorjahresmonat | 
|                     | (in Tausend) | in %                         | absolut (in Tausend) | in % |
|---|---:|---:|---:|---:|
| Langzeitarbeitslose     | 1.052 | 34,8 | 70 | 7,1 |
| davon Rechtskreis SGB III | 111   | 9,8  | 13 | 13,2 |
| Rechtskreis SGB II       | 941   | 49,9 | 57 | 6,4 |
| Quelle: Statistik der Bundesagentur für Arbeit |
""".strip()


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_defaults():
    base_dir = Path(__file__).resolve().parent
    defaults = {
        "model": "qwen/qwen2.5-vl-72b-instruct:free",
        "zoom": 5.0,
        "temperature": 1.0,
        "top_p": 1.0,
        # Large defaults lead to context-limit errors on many OpenRouter models; keep this modest.
        "max_tokens": 4096,
        "timeout": 45,
        "cache_db": str(base_dir / "pdf2markdown_cache.db"),
        "referer": os.environ.get("OPENROUTER_HTTP_REFERER", "https://localhost"),
        "title": os.environ.get("OPENROUTER_APP_TITLE", "pdf2markdown"),
        "sleep": 0.1,
        "retries": 3,
        "retry_delay": 5.0,
    }
    try:
        from fc_core import config as repo_config  # type: ignore
    except Exception:
        repo_config = None  # type: ignore
    if repo_config:
        defaults["model"] = getattr(repo_config, "QWEN_OCR_MODEL", defaults["model"])
        defaults["zoom"] = float(getattr(repo_config, "OCR_ZOOM", defaults["zoom"]))
        defaults["temperature"] = float(getattr(repo_config, "OCR_TEMPERATURE", defaults["temperature"]))
        defaults["top_p"] = float(getattr(repo_config, "OCR_TOP_P", defaults["top_p"]))
        defaults["max_tokens"] = int(getattr(repo_config, "OCR_MAX_TOKENS", defaults["max_tokens"]))
        defaults["timeout"] = int(getattr(repo_config, "OCR_TIMEOUT", defaults["timeout"]))
        defaults["referer"] = getattr(repo_config, "OPENROUTER_HTTP_REFERER", defaults["referer"]) if hasattr(repo_config, "OPENROUTER_HTTP_REFERER") else defaults["referer"]
        defaults["title"] = getattr(repo_config, "OPENROUTER_APP_TITLE", defaults["title"]) if hasattr(repo_config, "OPENROUTER_APP_TITLE") else defaults["title"]

    defaults["model"] = os.environ.get("QWEN_OCR_MODEL", defaults["model"])
    defaults["zoom"] = _float_env("OCR_ZOOM", float(defaults["zoom"]))
    defaults["temperature"] = _float_env("OCR_TEMPERATURE", float(defaults["temperature"]))
    defaults["top_p"] = _float_env("OCR_TOP_P", float(defaults["top_p"]))
    defaults["max_tokens"] = _int_env("OCR_MAX_TOKENS", int(defaults["max_tokens"]))
    defaults["timeout"] = _int_env("OCR_TIMEOUT", int(defaults["timeout"]))
    defaults["sleep"] = _float_env("PDF2MARKDOWN_SLEEP", float(defaults["sleep"]))
    defaults["retries"] = _int_env("OCR_RETRIES", int(defaults["retries"]))
    defaults["retry_delay"] = _float_env("OCR_RETRY_DELAY", float(defaults["retry_delay"]))
    return defaults


# --------------------
# Settings & Cache
# --------------------

@dataclass
class ConversionSettings:
    provider: str
    api_key: str
    model: str
    temperature: float
    top_p: float
    max_tokens: int
    timeout: int
    zoom: float
    prompt: str
    sleep: float
    max_parallel_requests: int
    retries: int
    retry_delay: float
    use_cache: bool
    cache_db: str
    referer: str
    title: str
    debug_image: bool


class SimpleCache:
    """
    Single-table cache with exactly three columns:
      filename TEXT NOT NULL,
      page_number INTEGER NOT NULL,
      content TEXT NOT NULL,
    PRIMARY KEY (filename, page_number)
    """

    def __init__(self, path: str, enabled: bool):
        self.path = path
        self.enabled = enabled
        self._conn: Optional[sqlite3.Connection] = None

    def __enter__(self) -> "SimpleCache":
        if self.enabled:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            self._ensure_schema()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_schema(self) -> None:
        if self._conn is None:
            return
        self._conn.execute("PRAGMA synchronous=FULL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                filename    TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                content     TEXT NOT NULL,
                PRIMARY KEY (filename, page_number)
            )
            """
        )
        self._conn.commit()

    def get(self, filename: str, page_number: int) -> Optional[str]:
        if self._conn is None:
            return None
        cur = self._conn.execute(
            "SELECT content FROM cache WHERE filename=? AND page_number=?",
            (filename, int(page_number)),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def put(self, filename: str, page_number: int, content: str) -> None:
        if self._conn is None:
            return
        self._conn.execute(
            "INSERT OR REPLACE INTO cache(filename, page_number, content) VALUES(?,?,?)",
            (filename, int(page_number), content or ""),
        )
        self._conn.commit()

    def bulk_get(self, filename: str, pages: Sequence[int]) -> Dict[int, str]:
        result: Dict[int, str] = {}
        if self._conn is None or not pages:
            return result
        placeholders = ",".join(["?"] * len(pages))
        q = f"SELECT page_number, content FROM cache WHERE filename=? AND page_number IN ({placeholders})"
        params: List[Any] = [filename] + [int(p) for p in pages]
        cur = self._conn.execute(q, params)
        for page_number, content in cur.fetchall():
            result[int(page_number)] = content
        return result


# --------------------
# Helpers
# --------------------

def _b64_data_uri_png(png: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def _b64_data_uri_pdf(pdf_bytes: bytes) -> str:
    return "data:application/pdf;base64," + base64.b64encode(pdf_bytes).decode("ascii")


def _page_to_png_bytes(page: Any, zoom: float) -> bytes:
    """
    Render a single page to PNG bytes. Used for OpenRouter models that expect images
    (data:image/png) instead of inline PDF data.
    """
    try:
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix.tobytes("png")
    except Exception:
        return b""


def _save_debug_image(data_bytes: bytes, pdf_path: Path, page_number: int, *, verbose: bool = False) -> None:
    """Persist the single-page PDF bytes for debugging (saves as .pdf).

    This previously saved PNGs; when debugging single-page PDFs we now write a .pdf file.
    """
    try:
        filename = f"{pdf_path.stem}-page-{page_number:04d}.pdf"
        output_path = Path.cwd() / filename
        output_path.write_bytes(data_bytes)
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to save debug PDF for page {page_number}: {exc}", file=sys.stderr)


def _trim_png_whitespace(
    png_bytes: bytes,
    *,
    verbose: bool = False,
    threshold: int = 12,
    padding: int = 4,
) -> bytes:
    if Image is None or ImageChops is None or ImageOps is None:
        if verbose:
            print("[WARN] Pillow not available; skipping whitespace trim", file=sys.stderr)
        return png_bytes
    try:
        with Image.open(io.BytesIO(png_bytes)) as image:
            image.load()
            working = image.convert("RGBA")
            rgb = working.convert("RGB")
            white_bg = Image.new("RGB", rgb.size, (255, 255, 255))
            diff = ImageChops.difference(rgb, white_bg).convert("L")
            diff = ImageOps.autocontrast(diff)
            if threshold > 0:
                diff = diff.point(lambda v: 255 if v > threshold else 0)  # type: ignore[arg-type]
            alpha_mask = working.getchannel("A") if "A" in working.getbands() else None
            if alpha_mask is not None:
                extrema = alpha_mask.getextrema()
                if (
                    isinstance(extrema, tuple)
                    and len(extrema) == 2
                    and isinstance(extrema[0], (int, float))
                    and isinstance(extrema[1], (int, float))
                    and extrema[0] < extrema[1]
                ):
                    alpha_mask = alpha_mask.point(lambda a: 255 if a > 0 else 0)  # type: ignore[arg-type]
                    diff = ImageChops.lighter(diff, alpha_mask)
            bbox = diff.getbbox()
            if not bbox:
                return png_bytes
            left, upper, right, lower = bbox
            if padding > 0:
                left = max(left - padding, 0)
                upper = max(upper - padding, 0)
                right = min(right + padding, working.width)
                lower = min(lower + padding, working.height)
            cropped = working.crop((left, upper, right, lower))
            with io.BytesIO() as buffer:
                cropped.save(buffer, format="PNG", optimize=True)
                return buffer.getvalue()
    except Exception as exc:
        if verbose:
            print(f"[WARN] Failed to trim whitespace: {exc}", file=sys.stderr)
    return png_bytes


def _fix_hyphenation(text: str) -> str:
    return text.replace("-\n", "").replace("\u00ad", "").replace("\u00AD", "")


def _dedupe_page_image_tags(text: str) -> str:
    lines = text.splitlines()
    seen: set[str] = set()
    deduped: List[str] = []
    for line in lines:
        key = line.strip()
        if key.startswith("# Image:"):
            if key in seen:
                continue
            seen.add(key)
        deduped.append(line)
    return "\n".join(deduped)


def _dedupe_global_image_tags(pages: Sequence[str]) -> str:
    seen: set[str] = set()
    merged: List[str] = []
    for page_text in pages:
        lines = page_text.splitlines()
        new_lines: List[str] = []
        for line in lines:
            key = line.strip()
            if key.startswith("# Image:"):
                if key in seen:
                    continue
                seen.add(key)
            new_lines.append(line)
        merged.append("\n".join(new_lines))
    return "\n\n".join(part for part in merged if part.strip())


def _clean_response_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.lower().startswith("```markdown"):
        cleaned = cleaned[len("```markdown"):].lstrip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()
    elif cleaned.startswith("```") and cleaned.endswith("```"):
        cleaned = cleaned[3:-3].strip()
    if cleaned.startswith('"') and cleaned.endswith('"') and len(cleaned) > 1:
        cleaned = cleaned[1:-1].strip()
    return cleaned


def _extract_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence):
        fragments: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                fragments.append(item.get("text", ""))
        return "\n".join(fragments)
    return ""


# --------------------
# Providers
# --------------------

def _get_openrouter_client(settings: ConversionSettings):
    if OpenAI is None:
        raise RuntimeError("The openai package is required. Install it with 'pip install openai'.")
    if not settings.api_key:
        raise RuntimeError("OpenRouter API key is not configured. Use --api-key or set OPENROUTER_API_KEY.")
    return OpenAI(
        api_key=settings.api_key,
        base_url=BASE_URL,
        default_headers={
            "HTTP-Referer": settings.referer,
            "X-Title": settings.title,
        },
    )


def _call_openrouter(client: Any, settings: ConversionSettings, image_bytes: bytes, mime_type: str = "application/pdf") -> str:
    # For OpenRouter we use data URIs. Some models only accept images, so we send PNG when requested.
    if mime_type == "image/png":
        data_uri = _b64_data_uri_png(image_bytes)
    else:
        data_uri = _b64_data_uri_pdf(image_bytes)
    last_exc: Optional[Exception] = None
    for attempt in range(1, settings.retries + 1):
        try:
            response = client.chat.completions.create(
                model=settings.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": settings.prompt},
                            # Send the single-page PDF as an inline data URI. OpenRouter clients accept image_url with a data URI.
                            {"type": "image_url", "image_url": {"url": data_uri}},
                        ],
                    }
                ],
                temperature=settings.temperature,
                top_p=settings.top_p,
                max_tokens=settings.max_tokens,
                timeout=settings.timeout,
            )
        except Exception as exc:
            last_exc = exc
            if attempt >= settings.retries:
                raise RuntimeError(f"OpenRouter API request failed after {attempt} attempt(s): {exc}") from exc
            if settings.retry_delay > 0:
                time.sleep(settings.retry_delay)
            continue
        if not response or not getattr(response, "choices", None):
            if attempt >= settings.retries:
                return ""
            if settings.retry_delay > 0:
                time.sleep(settings.retry_delay)
            continue
        message = response.choices[0].message
        content = getattr(message, "content", "")
        return _extract_message_content(content)
    if last_exc is not None:
        raise RuntimeError(f"OpenRouter API request failed: {last_exc}") from last_exc
    return ""


def _call_gemini(settings: ConversionSettings, image_bytes: bytes) -> str:
    if not settings.api_key:
        raise RuntimeError("Gemini API key is not configured. Use --api-key or set GEMINI_API_KEY.")
    # Send the bytes as base64-encoded data with application/pdf mime type
    data_uri = base64.b64encode(image_bytes).decode("ascii")
    url = f"{GEMINI_BASE_URL}/{settings.model}:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": settings.prompt},
                    {"inline_data": {"mime_type": "application/pdf", "data": data_uri}},
                ]
            }
        ],
        "generationConfig": {
            "temperature": settings.temperature,
            "topP": settings.top_p,
            "maxOutputTokens": settings.max_tokens,
            "thinkingConfig": {"thinkingBudget": GEMINI_THINKING_BUDGET},
        },
    }
    if "gemini" in settings.model.lower():
        payload["generationConfig"]["mediaResolution"] = "MEDIA_RESOLUTION_MEDIUM"

    last_exc: Optional[Exception] = None
    last_error: Optional[str] = None
    timeout_failures = 0
    max_timeout_failures = max(1, min(settings.retries, 3))
    using_fallback_key = settings.api_key == GEMINI_FALLBACK_KEY

    for attempt in range(1, settings.retries + 1):
        try:
            response = requests.post(
                url, params={"key": settings.api_key}, json=payload, timeout=settings.timeout
            )
        except requests.RequestException as exc:
            last_exc = exc
            if isinstance(exc, (requests_exceptions.Timeout, requests_exceptions.ReadTimeout, requests_exceptions.ConnectTimeout)):
                timeout_failures += 1
                if timeout_failures >= max_timeout_failures or attempt >= settings.retries:
                    hint = (
                        "Gemini fallback key is heavily rate limited; provide GEMINI_API_KEY or switch to --provider openrouter."
                        if using_fallback_key else
                        "Check your network connectivity or adjust --timeout/--retries."
                    )
                    raise RuntimeError(
                        f"Gemini API request timed out after {attempt} attempt(s) (timeout={settings.timeout}s). {hint}"
                    ) from exc
            if attempt >= settings.retries:
                raise RuntimeError(f"Gemini API request failed after {attempt} attempt(s): {exc}") from exc
            if settings.retry_delay > 0:
                time.sleep(settings.retry_delay)
            continue

        status = response.status_code
        if status == 429:
            try:
                body = response.json()
                msg = body.get("error", {}).get("message") or response.text
            except Exception:
                msg = response.text
            # Special-case: when using the free-tier quota limit message that contains
            # the substring "limit: 15", treat this as a transient rate-limit and
            # wait 30 seconds before retrying (up to settings.retries). This mirrors
            # existing retry logic but gives a longer backoff for the free-tier hard
            # limit message observed in practice.
            if "limit: 15" in (msg or ""):
                # If this was the last allowed attempt, raise the error; otherwise
                # sleep 30 seconds and continue to the next attempt.
                if attempt >= settings.retries:
                    raise RuntimeError(f"Gemini API error 429 (quota exhausted): {msg}")
                # Wait 30 seconds specifically for the free-tier quota message.
                time.sleep(30)
                continue
            raise RuntimeError(f"Gemini API error 429 (quota exhausted): {msg}")
        if status >= 500 or status == 408:
            last_error = f"Gemini API error {status}: {response.text}"
            if attempt >= settings.retries:
                raise RuntimeError(last_error)
            if settings.retry_delay > 0:
                time.sleep(settings.retry_delay)
            continue
        if status >= 400:
            raise RuntimeError(f"Gemini API error {status}: {response.text}")

        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            last_error = "Gemini API returned no candidates"
            if attempt >= settings.retries:
                raise RuntimeError(last_error)
            if settings.retry_delay > 0:
                time.sleep(settings.retry_delay)
            continue
        parts = candidates[0].get("content", {}).get("parts")
        fragments: List[str] = []
        if isinstance(parts, list):
            for part in parts:
                text_part = part.get("text") if isinstance(part, dict) else None
                if text_part:
                    fragments.append(text_part)
        if fragments:
            result = "\n".join(fragments).strip()
            return result

        last_error = "Gemini API returned no text parts"
        if attempt >= settings.retries:
            raise RuntimeError(last_error)
        if settings.retry_delay > 0:
            time.sleep(settings.retry_delay)

    if last_exc is not None:
        raise RuntimeError(f"Gemini API request failed: {last_exc}") from last_exc
    if last_error is not None:
        raise RuntimeError(last_error)
    return ""


def _is_quota_error(exc: Exception) -> bool:
    try:
        text = str(exc).lower()
    except Exception:
        return False
    return ("429" in text and "quota" in text) or "quota exhausted" in text or "resource_exhausted" in text


def _create_model_client(settings: ConversionSettings):
    if settings.provider == "openrouter":
        return _get_openrouter_client(settings)
    return None


# --------------------
# Core conversion
# --------------------

def convert_pdf_to_markdown(
    pdf_path: str,
    settings: ConversionSettings,
    *,
    on_page: Optional[Callable[[int, int, str, bool], None]] = None,
    progress_bar: bool = True,
    verbose: bool = False,
    pages: Optional[Sequence[int]] = None,
    include_page_headers: bool = True,
) -> str:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    doc = fitz.open(str(path))
    doc_total = doc.page_count
    if doc_total == 0:
        doc.close()
        return ""

    # Pages to process (1-based)
    if pages is not None:
        unique_pages: List[int] = []
        seen_pages: set[int] = set()
        for raw_page in pages:
            page_number = int(raw_page)
            if page_number < 1 or page_number > doc_total:
                doc.close()
                raise ValueError(f"Page number out of range: {page_number}")
            if page_number not in seen_pages:
                unique_pages.append(page_number)
                seen_pages.add(page_number)
        page_numbers = unique_pages
    else:
        page_numbers = list(range(1, doc_total + 1))

    selected_total = len(page_numbers)
    if selected_total == 0:
        doc.close()
        return ""

    client = _create_model_client(settings)

    def invoke_model(image_bytes: bytes, mime_type: str = "application/pdf") -> str:
        if settings.provider == "openrouter":
            return _call_openrouter(client, settings, image_bytes, mime_type)
        return _call_gemini(settings, image_bytes)

    out_texts: List[str] = ["" for _ in range(selected_total)]

    filename_key = path.name  # key = filename only, as requested
    initial_cached_map: Dict[int, str] = {}

    page_bar = None
    with SimpleCache(settings.cache_db, settings.use_cache) as cache:
        if cache and cache.enabled:
            initial_cached_map = cache.bulk_get(filename_key, page_numbers)

        initial_hits = len(initial_cached_map)

        # Progress bar shows cached count IMMEDIATELY (e.g., 50/80)
        if progress_bar and tqdm is not None:
            desc = "Processing pages" if pages is None else "Processing selected pages"
            page_bar = tqdm(total=selected_total, desc=desc, unit="page", initial=initial_hits)
            page_bar.refresh()  # force immediate render with initial count

        # Place cached pages into output right away (no progress update here)
        for idx, page_number in enumerate(page_numbers, start=1):
            cached = initial_cached_map.get(page_number)
            if cached is None:
                continue
            cleaned = _clean_response_text(cached)
            cleaned = _dedupe_page_image_tags(cleaned)
            final_page = _fix_hyphenation(cleaned)
            body_lines = final_page.splitlines()
            while body_lines and not body_lines[0].strip():
                body_lines.pop(0)
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            body = "\n".join(body_lines)
            # Only store the body (no header) in cache
            out_texts[idx - 1] = body
            if on_page is not None:
                try:
                    on_page(page_number, doc_total, body, True)
                except TypeError:
                    on_page(page_number, doc_total, body)  # type: ignore[misc]

        # Now schedule uncached pages
        max_parallel = max(1, settings.max_parallel_requests)
        executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=max_parallel) if max_parallel > 1 else None
        pending_futures: Dict[Future[str], Tuple[int, int]] = {}  # future -> (page_number, position_index)

        def submit_for_page(page_number: int, position_index: int) -> None:
            # Create a single-page PDF bytes containing only the requested page.
            # For OpenRouter we also render a PNG, because many models only accept images.
            single_pdf_bytes = b""
            png_bytes = b""
            try:
                new_doc = fitz.open()
                page = doc.load_page(page_number - 1)
                if settings.provider == "openrouter":
                    png_bytes = _page_to_png_bytes(page, settings.zoom)
                new_doc.insert_pdf(doc, from_page=page_number - 1, to_page=page_number - 1)
                with io.BytesIO() as buf:
                    new_doc.save(buf)
                    single_pdf_bytes = buf.getvalue()
                new_doc.close()
            except Exception as exc:
                if verbose:
                    print(f"[ERROR] Failed to build single-page PDF for page {page_number}: {exc}", file=sys.stderr)
                single_pdf_bytes = b""
            if settings.debug_image and single_pdf_bytes:
                _save_debug_image(single_pdf_bytes, path, page_number, verbose=bool(verbose))

            payload_bytes = png_bytes if settings.provider == "openrouter" and png_bytes else single_pdf_bytes
            payload_mime = "image/png" if settings.provider == "openrouter" and png_bytes else "application/pdf"
            if executor is None:
                # Synchronous
                try:
                    text = invoke_model(payload_bytes, payload_mime)
                except Exception as exc:
                    if _is_quota_error(exc):
                        raise
                    print(f"[ERROR] Failed to process page {page_number}: {exc}", file=sys.stderr)
                    text = ""
                finalize_uncached(page_number, position_index, text)
                if page_bar is not None:
                    page_bar.update(1)
                if settings.sleep > 0:
                    time.sleep(settings.sleep)
            else:
                future = executor.submit(invoke_model, payload_bytes, payload_mime)
                pending_futures[future] = (page_number, position_index)

        def finalize_uncached(page_number: int, position_index: int, text: str) -> None:
            cleaned = _clean_response_text(text or "")
            cleaned = _dedupe_page_image_tags(cleaned)
            final_page = _fix_hyphenation(cleaned)
            body_lines = final_page.splitlines()
            while body_lines and not body_lines[0].strip():
                body_lines.pop(0)
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            body = "\n".join(body_lines)
            # Only store the body (no header) in cache
            out_texts[position_index - 1] = body

            if cache and cache.enabled and body.strip():
                try:
                    cache.put(filename_key, page_number, body)
                except Exception as cache_exc:
                    if verbose:
                        print(f"[WARN] Cache store failed for page {page_number}: {cache_exc}", file=sys.stderr)

            if on_page is not None:
                try:
                    on_page(page_number, doc_total, body, False)
                except TypeError:
                    on_page(page_number, doc_total, body)  # type: ignore[misc]

        for idx, page_number in enumerate(page_numbers, start=1):
            if page_number in initial_cached_map:
                continue
            submit_for_page(page_number, idx)

        fatal_error: Optional[Exception] = None
        try:
            if executor is not None:
                while pending_futures:
                    done, _ = wait(list(pending_futures.keys()), return_when=FIRST_COMPLETED)
                    for fut in list(done):
                        page_number, position_index = pending_futures.pop(fut)
                        try:
                            text = fut.result()
                        except Exception as exc:
                            if _is_quota_error(exc):
                                fatal_error = exc
                                raise
                            print(f"[ERROR] Failed to process page {page_number}: {exc}", file=sys.stderr)
                            text = ""
                        finalize_uncached(page_number, position_index, text)
                        if page_bar is not None:
                            page_bar.update(1)
                        if settings.sleep > 0:
                            time.sleep(settings.sleep)
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            if fatal_error is not None:
                raise fatal_error

    # Cleanup
    doc.close()
    if page_bar is not None:
        page_bar.close()

    # Add headers only when assembling the final Markdown file (can be disabled with include_page_headers)
    final_texts_with_headers = []
    for idx, page_number in enumerate(page_numbers, start=1):
        body = out_texts[idx - 1]
        if include_page_headers:
            header = f'*** Page {page_number}/{doc_total} of "{path.name}" begins here ***'
            final_page = header if not body else f"{header}\n\n{body}"
        else:
            final_page = body
        final_texts_with_headers.append(final_page)
    final_text = _dedupe_global_image_tags(final_texts_with_headers)
    final_text = _fix_hyphenation(final_text)
    return final_text.strip()


# --------------------
# CLI
# --------------------

def _resolve_api_key(provider: str, cli_key: Optional[str]) -> str:
    if cli_key:
        return cli_key
    provider = provider.lower()
    if provider == "gemini":
        env_key = os.environ.get("GEMINI_API_KEY") or GEMINI_FALLBACK_KEY
        if env_key:
            return env_key
        raise RuntimeError("Gemini API key is not configured. Set GEMINI_API_KEY or use --api-key.")
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if not env_key:
        env_key = "xxxxxxxxxxxxxxxxx"
    if env_key:
        return env_key
    try:
        from fc_core import config as repo_config  # type: ignore
    except Exception:
        repo_config = None  # type: ignore
    if repo_config:
        cfg_key = getattr(repo_config, "OPENROUTER_API_KEY", None)
        if cfg_key:
            return cfg_key
    raise RuntimeError("OpenRouter API key is not configured. Set OPENROUTER_API_KEY or use --api-key.")


def _load_prompt(prompt_file: Optional[str]) -> str:
    if not prompt_file:
        return DEFAULT_PROMPT
    content = Path(prompt_file).read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return content


def main() -> int:
    defaults = _load_defaults()
    parser = argparse.ArgumentParser(
        description="Convert PDF files to Markdown using Gemini or OpenRouter vision models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              pdf2markdown input.pdf -o output.md
              pdf2markdown input.pdf --provider openrouter --model "qwen/qwen2.5-vl-72b-instruct:free" --verbose
              pdf2markdown report.pdf --prompt-file prompt.txt --cache-db ./data/cache.db
            """
        ).strip(),
    )
    parser.add_argument("input", help="Input PDF file path")
    parser.add_argument("-o", "--output", help="Output Markdown file path (default: input file name with .md extension)")
    parser.add_argument("--provider", choices=("gemini", "openrouter"), default="gemini", help="API provider to use (default: %(default)s)")
    parser.add_argument("--model", help="Model identifier for the selected provider (default: provider-specific)")
    parser.add_argument("--zoom", type=float, default=defaults["zoom"], help="Zoom factor for rendering pages (default: %(default).2f)")
    parser.add_argument("--temperature", type=float, default=defaults["temperature"], help="Sampling temperature (default: %(default)s)")
    parser.add_argument("--top-p", type=float, default=defaults["top_p"], help="Top-p nucleus sampling (default: %(default)s)")
    parser.add_argument("--max-tokens", type=int, default=defaults["max_tokens"], help="Maximum tokens for the response (default: %(default)s)")
    parser.add_argument("--timeout", type=int, default=defaults["timeout"], help="Request timeout in seconds (default: %(default)s)")
    parser.add_argument("--api-key", help="API key for the selected provider (overrides GEMINI_API_KEY or OPENROUTER_API_KEY)")
    parser.add_argument("--prompt-file", help="Path to a custom prompt text file")
    parser.add_argument("--cache-db", default=defaults["cache_db"], help="Path to the OCR cache database (default: %(default)s)")
    parser.add_argument("--sleep", type=float, default=defaults["sleep"], help="Delay between uncached API calls in seconds (default: %(default)s)")
    parser.add_argument("--retries", type=int, default=defaults["retries"], help="Maximum retry attempts for API calls (default: %(default)s)")
    parser.add_argument("--retry-delay", type=float, default=defaults["retry_delay"], help="Delay in seconds before retrying a failed API call (default: %(default)s)")
    parser.add_argument("--pages", help="Comma-separated list of 1-based page numbers to convert")
    parser.add_argument("--parallel", type=int, default=3, help="Maximum number of concurrent API requests (default: %(default)s)")
    parser.add_argument("--no-cache", dest="use_cache", action="store_false", help="Disable reuse of cached OCR results")
    parser.add_argument("--cache", dest="use_cache", action="store_true", help=argparse.SUPPRESS)
    parser.set_defaults(use_cache=True)
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--nopages", action="store_true", help="Do not include per-page headers in the output markdown")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging for each processed page")
    parser.add_argument("--debugimage", action="store_true", dest="debug_image", help="Persist rendered page PNGs next to the script for debugging")
    parser.add_argument("--referer", default=defaults["referer"], help="HTTP Referer header sent to OpenRouter (ignored for Gemini; default: %(default)s)")
    parser.add_argument("--title", default=defaults["title"], help="X-Title header sent to OpenRouter (ignored for Gemini; default: %(default)s)")

    args = parser.parse_args()

    # Parse page list
    pages_list: Optional[List[int]] = None
    if args.pages:
        try:
            raw_tokens = [token.strip() for token in args.pages.split(",") if token.strip()]
            if not raw_tokens:
                raise ValueError
            seen_pages: set[int] = set()
            pages_list = []
            for token in raw_tokens:
                page_number = int(token)
                if page_number <= 0:
                    raise ValueError
                if page_number not in seen_pages:
                    pages_list.append(page_number)
                    seen_pages.add(page_number)
        except ValueError:
            print(f"[ERROR] Invalid page list provided to --pages: {args.pages}", file=sys.stderr)
            return 1

    provider = args.provider.lower()
    model = args.model or ("gemini-flash-lite-latest" if provider == "gemini" else "qwen/qwen2.5-vl-72b-instruct:free")

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file does not exist: {input_path}", file=sys.stderr)
        return 1
    if input_path.suffix.lower() != ".pdf":
        print(f"[ERROR] Input file must have a .pdf extension: {input_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output) if args.output else input_path.with_suffix(".md")

    try:
        prompt = _load_prompt(args.prompt_file)
    except Exception as exc:
        print(f"[ERROR] Failed to load prompt: {exc}", file=sys.stderr)
        return 1

    try:
        api_key = _resolve_api_key(provider, args.api_key)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    settings = ConversionSettings(
        provider=provider,
        api_key=api_key,
        model=model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        zoom=args.zoom,
        prompt=prompt,
        sleep=max(args.sleep, 0.0),
        max_parallel_requests=max(1, args.parallel),
        retries=max(1, args.retries),
        retry_delay=max(0.0, args.retry_delay),
        use_cache=args.use_cache,
        cache_db=os.path.abspath(args.cache_db),
        referer=args.referer,
        title=args.title,
        debug_image=bool(getattr(args, "debug_image", False)),
    )

    def page_callback(idx: int, total: int, text: str, cache_hit: bool) -> None:
        if args.verbose:
            status = "cache" if cache_hit else "api"
            snippet = text.replace("\n", " ").strip()
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            print(f"[pdf2markdown] page {idx}/{total} ({status}): {len(text)} chars | {snippet}", file=sys.stderr)

    try:
        print(f"Converting {input_path} to {output_path}...")
        markdown_text = convert_pdf_to_markdown(
            str(input_path),
            settings,
            on_page=page_callback if args.verbose else None,
            progress_bar=not args.no_progress,
            verbose=args.verbose,
            pages=pages_list,
            include_page_headers=not getattr(args, "nopages", False),
        )
    except Exception as exc:
        print(f"[ERROR] Conversion failed: {exc}", file=sys.stderr)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown_text, encoding="utf-8")
    print(f"Successfully converted {input_path} to {output_path}")
    if args.verbose:
        print(f"[pdf2markdown] total characters: {len(markdown_text)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
