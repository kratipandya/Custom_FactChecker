# Custom Fact Checker (Citation Grounding + Evidence Localisation)

This repository contains a **custom fact-checking pipeline** used to *ground cited claims in a restricted corpus* (your local source PDFs). The core goal is **citation quality control**: ensure that quoted snippets are **actually present** in the cited source and (optionally) on the claimed page(s).

> **Attribution**  
> The **core pipeline logic** (including the evidence-localisation design and the PDF→Markdown conversion strategy with page markers) is attributed to **Prof. Heiko Neuhaus**.  
> Any adaptations in this repo (e.g., evaluation scripts, LLM-ensemble judge baseline, convenience wrappers, formatting changes) were added by the student for thesis evaluation and reproducibility.

---

## What this tool is (and is not)

### ✅ What it does well
- Enforces **evidence localisation**: checks whether an **expected snippet** exists in the source text.
- Supports **page-level traceability** when page markers are available.
- Produces **auditable diagnostic labels** (EXACT MATCH, WRONG PAGE, NOT FOUND, etc.).
- Acts as a **high-precision guardrail** against hallucinated/incorrect citations in a restricted corpus.

### ❌ What it does not guarantee
- An EXACT MATCH **does not automatically mean** the claim is logically *entailed* by the evidence.
- Lexical matching does **not** recognize paraphrases unless you provide a verbatim snippet (or use wildcards carefully).

---

## Pipeline overview

### Stage 1 — Source preparation (PDF → Markdown)
Each source PDF is converted into a Markdown file with **explicit page markers** (e.g., `Page 3/12`) so later checks can verify page-local evidence.

**Output:** one `.md` per source (page-marked)

### Stage 2 — Claim specification (`claims.csv`)
A structured CSV lists every cited claim, its source key, page reference, and a *verbatim* expected snippet.

**Output:** `claims.csv` (or similarly named file)

### Stage 3 — Verification (`factchecker.py`)
`factchecker.py` reads the claims file, loads the corresponding Markdown source, and verifies:
- whether the snippet exists, and
- whether it exists on the claimed page(s)

**Output:** `factcheck_report.txt` (detailed report + summary stats)

---

## File format: `claims.csv`

Required columns (recommended schema):

- `cite_key`  
  The LaTeX/BibTeX key for the source (used to map to the right `.md` file)

- `pages`  
  Page spec from the citation (supports: `p. 5`, `pp. 10–11`, `pp. 10, 16`, etc.)

- `claim_text`  
  Short natural-language description of the claim (the claim to verify)

- `context`  
  Where the claim appears (chapter/section/subsection)

- `expected_snippet`  
  **Verbatim** text copied from the source that supports the claim  
  (may include wildcard `*` to bridge formatting differences)

Optional but useful columns:
- `invalid` / `status_override` (if you maintain an INVALID workflow)
- `notes`

---

## Output report: status labels

The report assigns one of these (or similar) labels per claim:

- **EXACT MATCH** — snippet found on the referenced page(s)
- **WRONG PAGE** — snippet found in the file, but not on referenced page(s)
- **NOT FOUND IN ENTIRE DOCUMENT** — snippet never appears in the source
- **FILE MISSING** — expected `.md` source file missing
- **SNIPPET TOO SHORT** — snippet too short to be reliable
- **CLAIM NOT IN TEX** — claim not detected in the LaTeX chapter (optional check)
- **CONFIRMED INVALID** — explicitly marked invalid; should not be auto-fixed

---

## Rules (important)

1. **Expected snippet must be literal**  
   Don’t use paraphrases. Copy-paste directly from the source text.

2. **Snippet length matters**  
   Use at least ~5 words (or your configured minimum). Too short causes false matches.

3. **Preserve meaning, not formatting**  
   Minor whitespace/line-break differences are normal; the checker normalizes whitespace.

4. **Use wildcards sparingly**
   - `*` can bridge small gaps (e.g., hyphenation, line breaks)
   - overuse makes matches less precise

5. **Page markers must be reliable**
   If PDF→Markdown conversion loses text or page markers drift, you can get false negatives.

---

## How to run: evidence localisation check

### 1) Prepare sources
Convert each PDF into a page-marked Markdown file (one per source).  
Make sure your naming convention matches how `cite_key` maps to file paths.

### 2) Create / update `claims.csv`
Fill rows for each cited claim with `cite_key`, `pages`, `claim_text`, and `expected_snippet`.

### 3) Run the checker
Example:
```bash
python3 factchecker.py --claims claims.csv --sources ./sources_md --out factcheck_report.txt
