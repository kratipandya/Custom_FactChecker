# Citation Fact-Checker Documentation

## ⚠️ READ THIS FIRST - Common Critical Error

**NEVER put section names or context descriptions in the `expected_snippet` field!**

❌ **WRONG:**

```csv
neumannDigitalTransformation2025,pp.~10,"...","...","Section 1.2 – Digital divide lens"
```

✅ **CORRECT:**

```csv
neumannDigitalTransformation2025,pp.~10,"...","Section 1.2","large cities with dense internet-domain activity"
```

**The `expected_snippet` must be ACTUAL TEXT copied from the markdown file, not your section names!**

---

## Overview

The `factchecker.py` tool verifies that citations in your LaTeX document (`chapter.tex`) are properly supported by text snippets found in the corresponding markdown source files. This prevents AI hallucination by requiring exact textual evidence from specified pages.

## How to Use claims.csv

The tool reads citations from `claims.csv`, a comma-separated values file with five columns. Each row represents one citation to verify.

## How the Tool Works

The tool performs these steps:

1. **Reads `claims.csv`** - Loads all citations to verify
2. **For each citation:**
   - Uses `cite_key` to look up the corresponding markdown source file
   - Parses `pages` to determine which page(s) to check (e.g., `pp.~10, 16` → checks pages 10 and 16)
   - Searches for `expected_snippet` on those specific pages in the markdown file
   - Records whether the snippet was found (exact match/not found/wrong page)
3. **Generates report** - Creates `factcheck_report.txt` with detailed results

**What the tool does NOT do:**

- Does NOT parse or extract information from `chapter.tex`
- Does NOT automatically determine which section a citation is in
- Does NOT generate the `claim_text` or `context` fields
- These fields must be manually filled in the CSV for organizational purposes only

### CSV Format

```csv
cite_key,pages,claim_text,context,expected_snippet
```

### Column Descriptions

1. **`cite_key`** (required)

   - The citation key used in your LaTeX document
   - Must match exactly (e.g., `epFactsheet2025`, `oecdSkillsResilience2024`)
   - The tool uses this to locate the corresponding markdown source file
   - Example: `epFactsheet2025`

2. **`pages`** (required)

   - The page reference from your LaTeX citation
   - Should match the LaTeX format exactly, including tildes (`~`) and hyphens (`--`)
   - **Important:** Keep commas and spaces as they appear in LaTeX
   - Examples:
     - Single page: `p.~1`
     - Page range: `pp.~10--11` (checks pages 10 and 11)
     - Multiple pages: `pp.~10, 16` (checks pages 10 and 16)
     - Multiple ranges: `pp.~21--22, 30--31` (checks pages 21, 22, 30, 31)
     - Article reference: `Art.~145, p.~1`

3. **`claim_text`** (required)

   - A brief summary of what the citation claims in your chapter
   - Used only for reporting purposes - **NOT automatically extracted**
   - You must manually write this when creating the CSV
   - Keep it concise but descriptive
   - If it contains commas, enclose the entire field in double quotes
   - Example: `"Employability is a central focus of European Union social and economic policy"`

4. **`context`** (required)

   - Identifies where in your document this citation appears
   - Used only for organizing the report - **NOT automatically extracted**
   - You must manually enter this when creating the CSV (e.g., by section number)
   - Helps you locate citations when reviewing the report
   - Example: `Section 1: The German Model`
   - Example: `Section 2: Multi-Level Governance - German federalism`

5. **`expected_snippet`** (required, **MOST IMPORTANT**)

   - **MUST be EXACT text copied from the markdown source file**
   - **Character-for-character accuracy is required** - copy the actual words, punctuation, and formatting from the source
   - **This field contains TEXT FROM THE SOURCE DOCUMENT, not section names or descriptions**
   - **MINIMUM 5 WORDS REQUIRED** - Shorter snippets match randomly on wrong pages
   - Do NOT paraphrase, summarize, or rewrite - copy verbatim
   - Use wildcards (`*`) only to skip intervening words, not to change the actual text
   - **NEVER use `...` (three dots/ellipsis)** - if you need a wildcard, use `*` (asterisk)

   - **CRITICAL ERRORS TO AVOID:**

     - ✗ **FORBIDDEN**: `"text before ... text after"` - NEVER use `...` anywhere
     - ✗ **FORBIDDEN**: `"Section 1.2 ... employability"` - use `*` NOT `...`
     - ✓ **CORRECT**: `"text before * text after"` - use `*` for wildcards
     - ✗ **TOO SHORT**: `"institutionell erfasst"` (2 words) will match on wrong pages
     - ✗ **TOO SHORT**: `"Minijobs"` (1 word) will match everywhere
     - ✗ **NEVER put section names here**: `"Section 1.2 – Digital divide lens"` is WRONG
     - ✗ **NEVER put context descriptions**: `"Section 1.3 – German model"` is WRONG
     - ✗ **NEVER put your own summaries**: These must be ACTUAL QUOTES from the markdown
     - ✗ Writing `20–64` when the source says `aged 20 to 64`
     - ✗ Writing `resilient` when the source says `resilience`
     - ✗ Writing `pivotal` when the source uses `central` or `key`
     - ✓ **CORRECT**: Copy the exact phrase from the markdown: `aged 20 to 64 to 75%`
     - ✓ **CORRECT**: `The European employment strategy, dating back to 1997, established common objectives`

   - If the source says "European employment strategy", write `European employment strategy`, not `employment strategy`
   - If the source has commas, spaces, or specific wording, preserve them exactly
   - Should be distinctive enough to uniquely identify the supporting evidence
   - Keep it between 5-15 words typically but EXACT
   - **The tool will search ONLY on the specified page(s) first, then check other pages**

### Wildcard Matching with `*`

The `expected_snippet` field supports the Kleene star (`*`) wildcard for flexible matching:

- **Without wildcard:** `productivity` → matches exactly this word on the page
- **With wildcards:** `OECD * PISA * TALIS * PIAAC` → matches text containing these EXACT terms in order with any content in between (up to ~3000 characters)
- **Example:** `employment strategy * 1997 * established common objectives` will match text like:
  - "employment strategy, dating back to 1997, established common objectives"
  - "employment strategy introduced in 1997 that established common objectives"

**CRITICAL:** The parts between `*` wildcards must be EXACT QUOTES from the source:

- ✓ CORRECT: `employment strategy * 1997 * open method of coordination` (if the source contains these exact phrases)
- ✗ WRONG: `employment policy * 1997 * coordination` (if the source says "employment strategy" and "open method of coordination")

**How to create the expected_snippet:**

1. Open the markdown source file
2. Navigate to the specified page
3. Find the relevant text
4. **Copy the exact words** (not your summary)
5. If the quote is long, use `*` to connect the key parts: `first exact phrase * second exact phrase * third exact phrase`

Use wildcards when:

- The exact wording is long and you want to match key parts
- There are multiple exact terms that must appear in a specific order on the same page
- You need to skip less important words between key exact phrases

**REMINDER:** Only `*` (asterisk) is allowed as a wildcard. NEVER use `...` (ellipsis).

### CSV Formatting Rules

1. **No extra spaces** around commas (except inside quoted fields)
2. **Enclose fields in double quotes** if they contain commas
   - Example: `"Germany confronts a triple transformation of demography, digitalisation and decarbonisation"`
3. **Keep header row** as the first line
4. **One citation per row**
5. **Escape quotes** inside quoted fields by doubling them (`""`)

### Example CSV Rows

**✓ CORRECT Examples:**

```csv
cite_key,pages,claim_text,context,expected_snippet
epFactsheet2025,p.~1,Employability is a central focus of EU policy,Section 1: Introduction,The European employment strategy, dating back to 1997, established common objectives
eurostatEmployment2025,p.~2,Germany's employment rate is 81.3%,Section 1: German model,Germany (81.3%)
oecdSkillsResilience2024,p.~4,Resilience involves absorption and adaptation,Section 1: Framework,absorption, adaptation, and anticipation of the shock
```

**✗ WRONG Examples (DO NOT DO THIS):**

```csv
cite_key,pages,claim_text,context,expected_snippet
neumannDigitalTransformation2025,pp.~10,Digital divide affects regions,Section 1.2,Section 1.2 – Digital divide lens
                                                                                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                                                     WRONG! This is a section name, not text from the markdown!

iabForschungsbericht112022,pp.~21--22,Economic losses projected,Section 1.3,Section 1.3 – German model, war & energy shock
                                                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                                                             WRONG! This is a section name, not text from the markdown!
```

**Why the wrong examples fail:**

- The `expected_snippet` contains section names like "Section 1.2" or "Section 1.3"
- These are YOUR organizational labels, not text from the markdown source files
- The factchecker will search the markdown and find nothing matching "Section 1.2 – Digital divide lens"
- Result: "NOT FOUND IN ENTIRE DOCUMENT"

**How to fix:**

1. Open the markdown file `Digital transformation, employment change and the adaptation of regions in Germany.md`
2. Navigate to pages 10 and 16
3. Find the ACTUAL text about digital divide (e.g., "large cities with dense internet-domain activity")
4. Copy that ACTUAL text into `expected_snippet`
5. Keep the section name in the `context` field, NOT in `expected_snippet`

## Running the Fact-Checker

1. **Prepare your CSV:** Edit or create `claims.csv` with your citations
2. **Run the tool:** `python factchecker.py`
3. **Review the report:** Check `factcheck_report.txt` for results

## Understanding the Report

The tool generates a report with these match types:

- ✓ **EXACT MATCH** - Found exactly on the specified page
- ⚠ **WRONG PAGE** - Found in the document but on a different page than specified
- ⚠ **SNIPPET TOO SHORT** - Less than 5 words; will match randomly on wrong pages
- ✗ **NOT FOUND IN ENTIRE DOCUMENT** - The snippet was not found anywhere in the source document
- ? **FILE MISSING** - The markdown source file for this citation key was not found
- ⊗ **CONFIRMED INVALID** - The AI could not verify this citation and marked it as INVALID; these citations are acknowledged and should NOT be changed in automatic revisions - they will be addressed manually later

## AI Fix Workflow with INVALID Notation

When an AI attempts to fix failed citations in `claims.csv`, it has two options:

### Option 1: Fix the Citation

If the AI can find the correct snippet in the source:

- Update the `expected_snippet` field with the correct text
- Update the `pages` field if needed
- The citation will then pass verification

### Option 2: Mark as INVALID

If the AI cannot verify the citation (it's genuinely wrong/BS):

- Replace the `expected_snippet` with: **`INVALID`**
- This signals to you that the citation needs manual review
- The factchecker will report it as "⊗ MARKED INVALID"

**Example workflow:**

```csv
# Before AI fix attempt:
cite_key,pages,claim_text,context,expected_snippet
euSkillsUnion2025,p.~1,Green skills are important,Section 1,"Green skills are a cross-cutting topic"

# After AI fix - Scenario A (AI found the correct text):
euSkillsUnion2025,p.~1,Green skills are important,Section 1,"Green skills are among many much needed skills developed through VET"

# After AI fix - Scenario B (AI could not verify, marks as INVALID):
euSkillsUnion2025,p.~99,Something that doesn't exist,Section 1,INVALID
```

**What happens when you run factchecker.py:**

```
[5] Citation: euSkillsUnion2025 (page p.~99)
    ⊗ MARKED AS INVALID BY AI
    The AI could not verify this citation and marked it as INVALID.
    Action required: Manually review and fix this citation, or remove it if truly invalid.
```

**Your next steps for INVALID citations:**

1. Open the chapter and find where this citation appears
2. Determine if the claim is truly necessary
3. Either:
   - Find the correct supporting text in the source and update the CSV
   - Remove the citation from both the chapter and CSV
   - Replace with a different, verifiable citation

**The AI should NOT:**

- ✗ Provide suggestions or guesses for invalid citations
- ✗ Try to "fix" citations it can't verify
- ✗ Hallucinate alternative snippets

**The AI should:**

- ✓ Update `expected_snippet` with correct text if found in the source
- ✓ Write `INVALID` in `expected_snippet` if it cannot verify the claim
- ✓ Give up rather than guess

## Tips for AI

**⚠️ CRITICAL INSTRUCTION: The `expected_snippet` field MUST contain EXACT text copied verbatim from the markdown source file. Do NOT paraphrase, summarize, or rewrite. Copy the actual words character-by-character.**

**⚠️ NEVER PUT SECTION NAMES OR CONTEXT IN `expected_snippet`:**

- ✗ WRONG: `"Section 1.2 – Digital divide lens"`
- ✗ WRONG: `"Section 1.3 – German model, war & energy shock"`
- ✗ WRONG: Any text that starts with "Section"
- ✓ CORRECT: Actual text from the markdown file

When an AI generates citations for the LaTeX document:

1. **FIRST: Read the markdown source file** to find the exact text that supports your claim
2. **SECOND: Navigate to the specific page number** mentioned in the markdown file markers
3. **THIRD: Copy the exact words** from that page (do not paraphrase or use synonyms)
4. **FOURTH: Paste into `expected_snippet`** - this should be a direct quote from the markdown, NOT a section name
5. **Use wildcards (`*`) only to skip words**, not to change the actual text
6. **Verify page numbers** match exactly where you found the text
7. **Keep snippets short** but EXACT - 3-10 words of verbatim quotes
8. **Run the tool** after each batch of citations to verify accuracy
9. **Address failures** immediately - If you get "NOT FOUND", you likely:
   - Put a section name in `expected_snippet` instead of actual text
   - Paraphrased instead of copying exactly
   - Have the wrong page number
   - Are citing from the wrong markdown file

## Example Workflow

```bash
# 1. AI adds citations to chapter.tex
# 2. AI updates claims.csv with verification data
# 3. Run the fact-checker
python factchecker.py

# 4. Review the report
cat factcheck_report.txt

# 5. Fix any issues and re-run
python factchecker.py
```

This workflow ensures that every citation in your academic document is traceable to specific evidence in your source materials, preventing hallucination and maintaining scholarly integrity.
