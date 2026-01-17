#!/usr/bin/env python3
"""
Citation Fact-Checker for LaTeX Academic Documents

This tool verifies that citations in chapter.tex are supported by exact text
snippets found in the corresponding markdown source files at the specified pages.

It prevents AI hallucination by requiring exact textual evidence from specified pages.
"""

import re
import os
import csv
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Citation:
    """Represents a single citation to be verified."""
    cite_key: str
    pages: str
    claim_text: str
    context: str
    expected_snippet: str
    
    def is_marked_invalid(self) -> bool:
        """Check if this citation is marked as INVALID by AI."""
        return self.expected_snippet.strip().upper() == 'INVALID'


@dataclass
class VerificationResult:
    """Result of verifying a citation."""
    citation: Citation
    found: bool
    match_type: str  # 'exact', 'wrong_page', 'not_found', 'file_missing', 'marked_invalid', 'snippet_too_short', 'claim_not_in_tex'
    found_text: Optional[str] = None
    page_found: Optional[str] = None
    tex_claim_found: Optional[bool] = None  # Whether claim_text was found in chapter.tex


class FactChecker:
    """Main fact-checking engine."""
    
    # Mapping from citation keys to markdown filenames
    CITATION_TO_FILE = {
        'basicsEvs': 'basics_of_evs.md',
        'historyLinguistics': 'History_of_Lingusitics.md',
    }
    
    def __init__(self, markdown_dir: str = './markdown', tex_file: Optional[str] = None):
        self.markdown_dir = Path(markdown_dir)
        self.tex_file = Path(tex_file) if tex_file else None
        self.tex_content_normalized = None
        
        # Load and normalize tex file if provided
        if self.tex_file and self.tex_file.exists():
            with open(self.tex_file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Normalize whitespace for comparison
                self.tex_content_normalized = ' '.join(content.split()).lower()
        
    def parse_page_reference(self, page_ref: str) -> List[str]:
        """
        Parse LaTeX page references like 'p.~1', 'pp.~10--11', 'pp.~10, 16', 'Art.~145, p.~1'.
        Returns list of page numbers/identifiers to check.
        
        Supports:
        - Single page: p.~1
        - Page range: pp.~10--11 (pages 10, 11)
        - Page list: pp.~10, 16 or pp.~10 16 (pages 10, 16)
        - Multiple ranges: pp.~21--22, 30--31
        """
        pages = []
        
        # Remove LaTeX formatting
        cleaned = page_ref.replace('~', ' ').replace('\\', '')
        
        # Handle article references
        art_match = re.search(r'Art\.\s*(\d+)', cleaned)
        if art_match:
            pages.append(f"Art. {art_match.group(1)}")
        
        # Handle page ranges (pp. 10--11 or pp. 10-11)
        # This will find ALL ranges in the string
        range_matches = re.finditer(r'(\d+)\s*[-–—]+\s*(\d+)', cleaned)
        found_range = False
        for match in range_matches:
            found_range = True
            start, end = int(match.group(1)), int(match.group(2))
            for p in range(start, end + 1):
                if str(p) not in pages:  # Avoid duplicates
                    pages.append(str(p))
        
        # If no ranges found, extract all individual page numbers
        if not found_range:
            page_nums = re.findall(r'(?:pp?\.\s*)?(\d+)', cleaned)
            for num in page_nums:
                if num not in pages:  # Avoid duplicates
                    pages.append(num)
        
        return pages if pages else ['1']  # Default to page 1 if nothing found
    
    def load_markdown_pages(self, filename: str) -> Dict[str, str]:
        """
        Load a markdown file and split it by page markers.
        Returns dict mapping page numbers to page content.
        """
        filepath = self.markdown_dir / filename
        
        if not filepath.exists():
            return {}
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        pages = {}
        
        # Split by page markers like: *** Page 1/4 of "..." begins here ***
        page_pattern = r'\*\*\* Page (\d+(?:/\d+)?)(?: of ".*?")? begins here \*\*\*'
        splits = re.split(page_pattern, content)
        
        # splits will be: [content_before, page_num_1, content_1, page_num_2, content_2, ...]
        if len(splits) > 1:
            for i in range(1, len(splits), 2):
                if i + 1 < len(splits):
                    page_num = splits[i].split('/')[0]  # Extract "1" from "1/4"
                    page_content = splits[i + 1]
                    pages[page_num] = page_content
        else:
            # No page markers found, treat entire file as page 1
            pages['1'] = content
        
        return pages
    
    def check_claim_in_tex(self, claim_text: str) -> bool:
        """
        Check if the claim_text exists in the chapter.tex file.
        Normalizes whitespace for comparison (ignores line breaks, multiple spaces).
        Returns True if found, False otherwise.
        """
        if not self.tex_content_normalized:
            # No tex file loaded, skip check
            return True
        
        # Normalize the claim text (remove extra whitespace, convert to lowercase)
        claim_normalized = ' '.join(claim_text.split()).lower()
        
        # Check if the normalized claim appears in the normalized tex content
        return claim_normalized in self.tex_content_normalized
    
    def exact_match(self, needle: str, haystack: str) -> Tuple[bool, Optional[str]]:
        """
        Perform exact matching to find needle in haystack.
        Supports wildcard * notation (Kleene star) to match any text.
        Returns (found, matched_text).
        """
        # Normalize whitespace for comparison
        needle_norm = ' '.join(needle.split())
        haystack_norm = ' '.join(haystack.split())
        
        # Check if needle contains wildcards
        if '*' in needle_norm:
            # Convert wildcard pattern to regex
            # Escape special regex characters except *
            pattern_parts = []
            for part in needle_norm.split('*'):
                if part:
                    # Escape regex special chars and normalize
                    escaped = re.escape(part.lower())
                    pattern_parts.append(escaped)
            
            # Join with .{0,3000}? (non-greedy match for any characters, up to ~1 page)
            # This allows wildcards to match text spread across a full page
            pattern = r'.{0,3000}?'.join(pattern_parts)
            
            # Try to find the pattern
            match = re.search(pattern, haystack_norm.lower(), re.IGNORECASE | re.DOTALL)
            
            if match:
                return True, match.group(0)
            else:
                return False, None
        
        # Try exact match (case-insensitive)
        if needle_norm.lower() in haystack_norm.lower():
            # Find the actual matched text
            start_idx = haystack_norm.lower().find(needle_norm.lower())
            matched = haystack_norm[start_idx:start_idx + len(needle_norm)]
            return True, matched
        
        return False, None
    
    def verify_citation(self, citation: Citation, debug=False) -> VerificationResult:
        """Verify a single citation against its source markdown file."""
        
        # FIRST: Check if claim_text exists in chapter.tex
        claim_in_tex = self.check_claim_in_tex(citation.claim_text)
        
        # Check if citation is marked as INVALID by AI
        if citation.is_marked_invalid():
            return VerificationResult(
                citation=citation,
                found=False,
                match_type='marked_invalid',
                tex_claim_found=claim_in_tex,
            )
        
        # If claim_text not found in tex, flag it but continue checking
        if not claim_in_tex:
            # Still check the snippet, but flag the claim_text issue
            pass
        
        # REJECT snippets using "..." - only "*" wildcard is allowed
        if '...' in citation.expected_snippet:
            return VerificationResult(
                citation=citation,
                found=False,
                match_type='invalid_ellipsis',
                tex_claim_found=claim_in_tex,
            )
        
        # Validate minimum snippet length (at least 5 words)
        snippet_word_count = len(citation.expected_snippet.split())
        if snippet_word_count < 5:
            # Too short! Will match randomly on wrong pages
            return VerificationResult(
                citation=citation,
                found=False,
                match_type='snippet_too_short',
                tex_claim_found=claim_in_tex,
            )
        
        # Get the markdown filename
        filename = self.CITATION_TO_FILE.get(citation.cite_key)
        
        if not filename:
            return VerificationResult(
                citation=citation,
                found=False,
                match_type='file_missing',
                tex_claim_found=claim_in_tex,
            )
        
        # Load pages from markdown
        pages = self.load_markdown_pages(filename)
        
        if not pages:
            return VerificationResult(
                citation=citation,
                found=False,
                match_type='file_missing',
                tex_claim_found=claim_in_tex,
            )
        
        # DEBUG
        # print(f"\n[DEBUG] Verifying citation: {citation.cite_key}")
        # print(f"[DEBUG] Available pages in markdown: {list(pages.keys())}")
        
        # Parse which pages to check
        pages_to_check = self.parse_page_reference(citation.pages)
        # print(f"[DEBUG] Pages to check from '{citation.pages}': {pages_to_check}")
        
        # Check each page for the expected snippet
        for page_num in pages_to_check:
            if page_num not in pages:
                # print(f"[DEBUG] Page {page_num} not in pages dict!")
                continue
            
            page_content = pages[page_num]
            # print(f"[DEBUG] Checking page {page_num}, content length: {len(page_content)}")
            # print(f"[DEBUG] Expected snippet: {citation.expected_snippet[:80]}...")
            
            # Try exact match
            found, matched_text = self.exact_match(
                citation.expected_snippet,
                page_content
            )
            # print(f"[DEBUG] Match result: found={found}")
            
            if found:
                # Check if we should flag claim_text mismatch even when snippet is found
                if not claim_in_tex:
                    return VerificationResult(
                        citation=citation,
                        found=True,
                        match_type='claim_not_in_tex',
                        found_text=matched_text,
                        page_found=page_num,
                        tex_claim_found=False,
                    )
                else:
                    return VerificationResult(
                        citation=citation,
                        found=True,
                        match_type='exact',
                        found_text=matched_text,
                        page_found=page_num,
                        tex_claim_found=True,
                    )
        
        # Not found on any specified page
        # Try to find it on other pages (wrong page reference?)
        for page_num, page_content in pages.items():
            if page_num not in pages_to_check:
                found, matched_text = self.exact_match(
                    citation.expected_snippet,
                    page_content
                )
                if found:
                    return VerificationResult(
                        citation=citation,
                        found=True,
                        match_type='wrong_page',
                        found_text=matched_text,
                        page_found=page_num,
                        tex_claim_found=claim_in_tex,
                    )
        
        return VerificationResult(
            citation=citation,
            found=False,
            match_type='not_found',
            tex_claim_found=claim_in_tex,
        )
    
    def generate_report(self, results: List[VerificationResult]) -> str:
        """Generate a detailed human-readable report."""
        
        lines = []
        lines.append("=" * 80)
        lines.append("CITATION FACT-CHECK REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        # Summary statistics
        total = len(results)
        exact_matches = sum(1 for r in results if r.match_type == 'exact')
        claim_not_in_tex = sum(1 for r in results if r.match_type == 'claim_not_in_tex')
        not_found = sum(1 for r in results if r.match_type == 'not_found')
        wrong_page = sum(1 for r in results if r.match_type == 'wrong_page')
        file_missing = sum(1 for r in results if r.match_type == 'file_missing')
        marked_invalid = sum(1 for r in results if r.match_type == 'marked_invalid')
        snippet_too_short = sum(1 for r in results if r.match_type == 'snippet_too_short')
        invalid_ellipsis = sum(1 for r in results if r.match_type == 'invalid_ellipsis')
        
        lines.append("SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total citations checked: {total}")
        lines.append(f"  ✓ Exact matches:       {exact_matches} ({exact_matches/total*100:.1f}%)")
        lines.append(f"  ✗ Claim not in .tex:   {claim_not_in_tex} ({claim_not_in_tex/total*100:.1f}%) - FAILURE")
        lines.append(f"  ⚠ Wrong page:          {wrong_page} ({wrong_page/total*100:.1f}%)")
        lines.append(f"  ✗ Not found:           {not_found} ({not_found/total*100:.1f}%)")
        lines.append(f"  ⚠ Snippet too short:   {snippet_too_short} ({snippet_too_short/total*100:.1f}%)")
        lines.append(f"  ✗ Invalid ellipsis:    {invalid_ellipsis} ({invalid_ellipsis/total*100:.1f}%)")
        lines.append(f"  ? File missing:        {file_missing} ({file_missing/total*100:.1f}%)")
        lines.append(f"  ⊗ Confirmed INVALID:   {marked_invalid} ({marked_invalid/total*100:.1f}%) - DO NOT auto-fix")
        lines.append("")
        
        # Detailed results
        lines.append("DETAILED RESULTS")
        lines.append("=" * 80)
        lines.append("")
        
        # Add note about INVALID citations
        if marked_invalid > 0:
            lines.append("NOTE: Citations marked as 'INVALID' are acknowledged as unverifiable.")
            lines.append("      These should NOT be changed in automatic revisions.")
            lines.append("      They will be addressed manually at a later stage.")
            lines.append("")
            lines.append("-" * 80)
            lines.append("")
        
        # Helper function to add claim_text warning
        def add_claim_warning(lines_list, result):
            """Add warning about claim_text not found in chapter.tex if applicable."""
            if result.tex_claim_found is False:
                lines_list.append("")
                lines_list.append("    " + "=" * 76)
                lines_list.append("    ⚠️ ADDITIONAL WARNING: CLAIM TEXT NOT FOUND IN chapter.tex")
                lines_list.append("    " + "=" * 76)
                lines_list.append("    Your CSV and LaTeX document are OUT OF SYNC!")
                lines_list.append("")
                lines_list.append("    🔍 This claim_text from CSV does NOT exist in chapter.tex:")
                claim_preview = result.citation.claim_text[:200]
                if len(result.citation.claim_text) > 200:
                    claim_preview += "..."
                lines_list.append(f"    \"{claim_preview}\"")
                lines_list.append("")
                lines_list.append("    📋 ACTION REQUIRED:")
                lines_list.append(f"    1. Search for citation {{" + result.citation.cite_key + "}} in chapter.tex")
                lines_list.append("    2. Copy the EXACT sentence/paragraph from chapter.tex")
                lines_list.append("    3. Update claim_text in claims.csv to match exactly")
                lines_list.append("    4. Keep all LaTeX commands (\\enquote, \\parencite, etc.) intact")
        
        for i, result in enumerate(results, 1):
            lines.append(f"[{i}] Citation: {result.citation.cite_key} (page {result.citation.pages})")
            lines.append(f"    Claim: {result.citation.claim_text}")
            lines.append(f"    Expected snippet: \"{result.citation.expected_snippet}\"")
            lines.append("")
            
            if result.match_type == 'exact':
                lines.append(f"    ✓ EXACT MATCH FOUND on page {result.page_found}")
                if result.found_text:
                    lines.append(f"    Found text: \"{result.found_text}\"")
                lines.append("    Action required: None - citation is correct, do not change")
                add_claim_warning(lines, result)
            
            elif result.match_type == 'claim_not_in_tex':
                lines.append(f"    ⚠ SNIPPET FOUND BUT CLAIM TEXT NOT IN chapter.tex!")
                lines.append(f"    ✓ The expected snippet WAS found on page {result.page_found}")
                if result.found_text:
                    lines.append(f"    Found text: \"{result.found_text}\"")
                lines.append("")
                lines.append("    ⚠ HOWEVER: The claim_text from your CSV does NOT exist in chapter.tex")
                lines.append("    This means your CSV and your LaTeX document are OUT OF SYNC!")
                lines.append("")
                lines.append("    🔍 Claim text that should be in chapter.tex:")
                # Show first 200 chars of claim
                claim_preview = result.citation.claim_text[:200]
                if len(result.citation.claim_text) > 200:
                    claim_preview += "..."
                lines.append(f"    \"{claim_preview}\"")
                lines.append("")
                lines.append("    💡 POSSIBLE CAUSES:")
                lines.append("    1. You edited chapter.tex but didn't update the CSV")
                lines.append("    2. The claim_text in CSV has typos or different wording")
                lines.append("    3. LaTeX commands differ (e.g., \\enquote vs quotes, \\parencite format)")
                lines.append("    4. Whitespace/line breaks are different")
                lines.append("")
                lines.append("    📋 ACTION REQUIRED:")
                lines.append("    1. Search for this citation in chapter.tex: {" + result.citation.cite_key + "}")
                lines.append("    2. Copy the EXACT sentence/paragraph from chapter.tex")
                lines.append("    3. Update the claim_text field in claims.csv to match chapter.tex exactly")
                lines.append("    4. Keep all LaTeX commands (\\enquote, \\parencite, etc.) intact")
                lines.append("")
                lines.append("    Note: The snippet verification passed, so your citation IS valid.")
                lines.append("          This is just a CSV maintenance issue to keep them synchronized.")
            
            elif result.match_type == 'wrong_page':
                lines.append("    ⚠ FOUND ON WRONG PAGE!")
                lines.append(f"    Expected page: {result.citation.pages}")
                lines.append(f"    Actually found on page: {result.page_found}")
                if result.found_text:
                    lines.append(f"    Found text: \"{result.found_text}\"")
                add_claim_warning(lines, result)
            
            elif result.match_type == 'not_found':
                lines.append("    ✗ NOT FOUND IN ENTIRE DOCUMENT")
                lines.append("    The expected snippet was not found on the specified page(s) or anywhere else in the source document.")
                
                # Add special debugging info for wildcards
                if '*' in result.citation.expected_snippet:
                    lines.append("")
                    lines.append("    ⚠ WILDCARD PATTERN DEBUGGING:")
                    lines.append("    Your snippet contains wildcards (*). Remember:")
                    lines.append("    • Text BETWEEN wildcards must match EXACTLY (word-for-word, including punctuation)")
                    lines.append("    • The pattern is looking for these exact text fragments:")
                    
                    # Split by * and show what needs to match exactly
                    parts = result.citation.expected_snippet.split('*')
                    
                    # Get the correct filename from the mapping
                    filename = self.CITATION_TO_FILE.get(result.citation.cite_key)
                    source_path = self.markdown_dir / filename if filename else None
                    
                    # Try to find which fragments actually exist in the source
                    problematic_fragments = []
                    
                    if source_path and source_path.exists():
                        try:
                            with open(source_path, 'r', encoding='utf-8') as f:
                                full_content = f.read()
                                # Normalize whitespace like we do in exact_match
                                normalized_content = ' '.join(full_content.split()).lower()
                                
                                for i, part in enumerate(parts):
                                    if part.strip():
                                        lines.append(f"      [{i+1}] \"{part}\"")
                                        
                                        # Check if this fragment exists in the document at all
                                        normalized_fragment = ' '.join(part.split()).lower()
                                        if normalized_fragment not in normalized_content:
                                            problematic_fragments.append((i+1, part.strip()))
                        except Exception:
                            # If file reading fails, just show fragments without analysis
                            for i, part in enumerate(parts):
                                if part.strip():
                                    lines.append(f"      [{i+1}] \"{part}\"")
                    else:
                        # File doesn't exist, just show fragments
                        for i, part in enumerate(parts):
                            if part.strip():
                                lines.append(f"      [{i+1}] \"{part}\"")
                    
                    lines.append("")
                    
                    # Add automated hints for fragments that don't exist at all
                    if problematic_fragments:
                        lines.append("    ⚠️ AUTOMATIC HINT DETECTED:")
                        for idx, fragment in problematic_fragments:
                            lines.append(f"      Fragment [{idx}] does NOT appear anywhere in the source document!")
                            lines.append(f"      The text \"{fragment}\" cannot be found.")
                            
                            # Suggest trying without the first word
                            words = fragment.split()
                            if len(words) > 1:
                                suggested = ' '.join(words[1:])
                                lines.append(f"      Try removing the first word: \"{suggested}\"")
                        lines.append("")
                    
                    # Add context-aware preview when all fragments exist but pattern doesn't match
                    elif source_path and source_path.exists() and len(parts) >= 2:
                        try:
                            with open(source_path, 'r', encoding='utf-8') as f:
                                full_content = f.read()
                                normalized_content = ' '.join(full_content.split()).lower()
                                
                                # Check if all fragments exist (strip whitespace from fragments first)
                                all_exist = True
                                fragment_positions = []
                                stripped_parts = [p.strip() for p in parts if p.strip()]
                                
                                for part in stripped_parts:
                                    normalized_fragment = ' '.join(part.split()).lower()
                                    pos = normalized_content.find(normalized_fragment)
                                    if pos == -1:
                                        all_exist = False
                                        break
                                    fragment_positions.append((normalized_fragment, pos))
                                
                                # If all fragments exist and they're reasonably close (within 500 chars)
                                if all_exist and len(fragment_positions) >= 2:
                                    min_pos = min(p[1] for p in fragment_positions)
                                    max_pos = max(p[1] + len(p[0]) for p in fragment_positions)
                                    
                                    if max_pos - min_pos <= 500:
                                        lines.append("    💡 CONTEXT-AWARE HINT:")
                                        lines.append("    All fragments exist in the document, but the pattern doesn't match.")
                                        lines.append("    Here's what the source actually looks like:")
                                        lines.append("")
                                        
                                        # Extract context around the fragments (with padding)
                                        context_start = max(0, min_pos - 50)
                                        context_end = min(len(normalized_content), max_pos + 50)
                                        context = normalized_content[context_start:context_end]
                                        
                                        # Show the context with visual markers
                                        lines.append(f"    Source text: \"...{context}...\"")
                                        lines.append("")
                                        
                                        # Highlight what's between the fragments
                                        first_frag = fragment_positions[0][0]
                                        last_frag = fragment_positions[-1][0]
                                        first_pos_in_context = context.find(first_frag)
                                        last_pos_in_context = context.find(last_frag)
                                        
                                        if first_pos_in_context != -1 and last_pos_in_context != -1:
                                            between_start = first_pos_in_context + len(first_frag)
                                            between_text = context[between_start:last_pos_in_context]
                                            
                                            if between_text:
                                                lines.append(f"    Text BETWEEN your wildcards: \"{between_text.strip()}\"")
                                                lines.append("")
                                                
                                                # Give actionable guidance
                                                word_count = len(between_text.strip().split())
                                                lines.append(f"    💡 The text between wildcards is {word_count} words long.")
                                                lines.append("    💡 SUGGESTED FIX:")
                                                lines.append(f"       • If the between-text is too long, add more specific wildcards")
                                                lines.append(f"       • If fragments have extra words at the start/end, trim them")
                                                lines.append(f"       • Copy the exact text from the source, including all punctuation")
                                                lines.append("")
                                                lines.append("    ⚠️ IMPORTANT:")
                                                lines.append("       → You MUST use a DIFFERENT snippet from the source document NOW!")
                                                lines.append("       → Wildcards allow flexibility - find another quote that works")
                                                lines.append("       → Don't try to fix this problematic snippet")
                                        
                                        lines.append("")
                        except Exception:
                            pass  # Silently fail if context preview doesn't work
                    
                    lines.append("    Common issues:")
                    lines.append("    • Wrong punctuation: \"China,\" vs \"China;\" or \"China.\"")
                    lines.append("    • Leading/trailing words: \" and increased\" should be \"increased\"")
                    lines.append("    • Missing words: Make sure each fragment appears verbatim in the source")
                    lines.append("")
                    lines.append("    NOTE: Whitespace (spaces, newlines, tabs) is normalized automatically.")
                    lines.append("          Multiple spaces/newlines become single spaces during matching.")
                    lines.append("")
                    lines.append("    TIP: If a fragment won't match, try removing leading/trailing words")
                    lines.append("         Example: \"China * and increased\" → \"China * increased\"")
                
                lines.append("")
                lines.append("    Action required:")
                lines.append("      1. Verify the citation exists in the correct source file")
                lines.append("      2. Check the page number is correct")
                lines.append("      3. Copy the EXACT text from the markdown (not a paraphrase)")
                lines.append("      4. If the citation cannot be verified, mark expected_snippet as 'INVALID'")
                add_claim_warning(lines, result)
            
            elif result.match_type == 'snippet_too_short':
                lines.append("    ⚠ SNIPPET TOO SHORT (< 5 words)")
                lines.append(f"    Word count: {len(result.citation.expected_snippet.split())} words")
                lines.append("    Snippets with fewer than 5 words will match randomly on wrong pages.")
                lines.append("    Action required:")
                lines.append("      1. Expand snippet to at least 5 words by adding more context from the source")
                lines.append("      2. Or mark expected_snippet as 'INVALID' if you cannot find verifiable text")
                add_claim_warning(lines, result)
            
            elif result.match_type == 'invalid_ellipsis':
                lines.append("    ✗ INVALID ELLIPSIS (...) DETECTED")
                lines.append("    ERROR: The expected_snippet contains '...' which is NOT ALLOWED.")
                lines.append("    Action required:")
                lines.append("      1. If you need a wildcard, use * (asterisk) NOT ... (three dots)")
                lines.append("      2. NEVER use ... anywhere in expected_snippet")
                lines.append("      3. Copy the EXACT text from the source, or use * for wildcards")
                lines.append("      4. Example: 'text before * text after' NOT 'text before ... text after'")
                add_claim_warning(lines, result)
            
            elif result.match_type == 'file_missing':
                lines.append("    ? FILE MISSING")
                lines.append(f"    The markdown source file for '{result.citation.cite_key}' was not found.")
                lines.append("    Action required: Add source file to markdown directory.")
                add_claim_warning(lines, result)
            
            elif result.match_type == 'marked_invalid':
                lines.append("    ⊗ CONFIRMED INVALID (AI could not verify)")
                lines.append("    This citation has been marked as INVALID and acknowledged.")
                lines.append("    → DO NOT CHANGE in automatic revisions - will be addressed manually later.")
                lines.append("    Status: Waiting for manual review and resolution.")
                add_claim_warning(lines, result)
            
            lines.append("")
            lines.append("-" * 80)
            lines.append("")
        
        return "\n".join(lines)


def load_citations_from_csv(csv_file: str) -> List[Citation]:
    """
    Load citations from a CSV file.
    
    CSV format should be:
    cite_key,pages,claim_text,context,expected_snippet
    
    Example:
    epFactsheet2025,p.~1,"Employability is central...","Section 1","employment strategy"
    
    The expected_snippet field supports wildcards (*) for flexible matching.
    Example: "OECD * PISA * TALIS * PIAAC" will match text with those terms in order.
    """
    citations = []
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"Warning: CSV file '{csv_file}' not found. Using hardcoded citations.")
        return None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            citation = Citation(
                cite_key=row['cite_key'].strip(),
                pages=row['pages'].strip(),
                claim_text=row['claim_text'].strip(),
                context=row['context'].strip(),
                expected_snippet=row['expected_snippet'].strip()
            )
            citations.append(citation)
    
    return citations


def parse_tex_citations(tex_file: str) -> Dict[str, int]:
    """
    Parse a .tex file and count citations in each \\section, \\subsection, and \\chapter.
    Each section counts only the citations directly within it (not including subsections).
    
    Returns a dict mapping section names to citation counts.
    """
    tex_path = Path(tex_file)
    if not tex_path.exists():
        print(f"Warning: TeX file '{tex_file}' not found.")
        return {}
    
    with open(tex_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all sections/subsections/chapters with their content
    # Pattern: \section{...}, \section*{...}, \subsection{...}, \subsection*{...}, \chapter{...}, \chapter*{...}
    section_pattern = r'\\(chapter|section|subsection)\*?\{([^}]+)\}'
    
    sections = []
    for match in re.finditer(section_pattern, content):
        level = match.group(1)  # chapter, section, or subsection
        title = match.group(2)
        start_pos = match.end()
        sections.append({
            'level': level,
            'title': title,
            'start': start_pos,
            'index': len(sections)
        })
    
    # For each section, find citations until the NEXT section at ANY level (to avoid double-counting)
    citation_counts = {}
    
    for i, section in enumerate(sections):
        # Find end position: next section at ANY level (not cumulative)
        end_pos = len(content)
        
        if i + 1 < len(sections):
            end_pos = sections[i + 1]['start']
        
        section_content = content[section['start']:end_pos]
        
        # Count citations: \cite[...]{...}, \parencite[...]{...}, or other cite commands
        cite_pattern = r'\\(?:paren)?cite(?:\[[^\]]*\])?\{[^}]+\}'
        citations = re.findall(cite_pattern, section_content)
        
        # Format the section name with hierarchy indicator
        level_indicator = {
            'chapter': '\\chapter',
            'section': '\\section',
            'subsection': '\\subsection'
        }
        
        section_key = f"{level_indicator[section['level']]}*{{{section['title']}}}"
        citation_counts[section_key] = len(citations)
    
    return citation_counts


def print_documentation_with_citation_summary(tex_file: Optional[str] = None):
    """
    Print the factchecker.md documentation, optionally preceded by citation counts from a .tex file.
    """
    # First, print citation summary if tex file provided
    if tex_file:
        sys.stdout.flush()  # Flush any buffered output first
        
        # Use buffer.write for UTF-8 encoding on Windows
        sys.stdout.buffer.write(("=" * 80 + "\n").encode('utf-8'))
        sys.stdout.buffer.write("CITATION SUMMARY BY SECTION\n".encode('utf-8'))
        sys.stdout.buffer.write(("=" * 80 + "\n").encode('utf-8'))
        sys.stdout.buffer.write("\n".encode('utf-8'))
        sys.stdout.flush()
        
        citation_counts = parse_tex_citations(tex_file)
        
        if citation_counts:
            total_citations = sum(citation_counts.values())
            sys.stdout.buffer.write(f"Total citations found: {total_citations}\n".encode('utf-8'))
            sys.stdout.buffer.write("\n".encode('utf-8'))
            
            for section_name, count in citation_counts.items():
                line = f"{section_name}: {count} citation{'s' if count != 1 else ''}\n"
                sys.stdout.buffer.write(line.encode('utf-8'))
            
            sys.stdout.buffer.write("\n".encode('utf-8'))
            sys.stdout.buffer.write(("=" * 80 + "\n").encode('utf-8'))
            sys.stdout.buffer.write("\n".encode('utf-8'))
            sys.stdout.flush()
        else:
            sys.stdout.buffer.write(f"No sections found in {tex_file}\n".encode('utf-8'))
            sys.stdout.flush()
            print(flush=True)
    
    # Now print the factchecker.md documentation
    doc_path = Path(__file__).parent / 'factchecker.md'
    
    if not doc_path.exists():
        print("Warning: factchecker.md not found. Skipping documentation output.")
        return
    
    print("=" * 80, flush=True)
    print("FACTCHECKER DOCUMENTATION", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)
    
    sys.stdout.flush()  # Flush before writing to buffer
    
    with open(doc_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Print to stdout with UTF-8 encoding
    # Use sys.stdout.buffer to write UTF-8 directly on Windows
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout.buffer.write(content.encode('utf-8'))
        sys.stdout.buffer.write(b'\n')
        sys.stdout.buffer.flush()
    else:
        print(content)


def main():
    """Main execution function. Reads citations from CSV or uses hardcoded ones."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Citation Fact-Checker: Verify citations against markdown sources'
    )
    parser.add_argument(
        '--prompt', '-prompt',
        dest='prompt_file',
        metavar='TEX_FILE',
        help='Display citation count summary from a .tex file, then show documentation'
    )
    
    args = parser.parse_args()
    
    # If --prompt is specified, print documentation with citation summary and exit
    if args.prompt_file:
        print_documentation_with_citation_summary(args.prompt_file)
        return 0
    
    # Normal fact-checking operation
    # Try to load from CSV first
    csv_file = 'claims.csv'
    citations_to_check = load_citations_from_csv(csv_file)
    
    # If CSV not found, use hardcoded citations (fallback)
    if citations_to_check is None:
        citations_to_check = get_hardcoded_citations()
    
    # Initialize fact checker with optional chapter.tex for claim verification
    tex_file = 'chapter.tex'
    tex_path = Path(tex_file)
    if tex_path.exists():
        print(f"Loading {tex_file} for claim_text verification...")
        checker = FactChecker(markdown_dir='./markdown', tex_file=tex_file)
    else:
        print(f"Warning: {tex_file} not found. Skipping claim_text verification.")
        checker = FactChecker(markdown_dir='./markdown')
    
    # Verify all citations
    print("Starting fact-check process...")
    print(f"Checking {len(citations_to_check)} citations...")
    print()
    
    results = []
    for i, citation in enumerate(citations_to_check, 1):
        print(f"[{i}/{len(citations_to_check)}] Checking {citation.cite_key} (page {citation.pages})...")
        result = checker.verify_citation(citation)
        results.append(result)
    
    print()
    print("Fact-check complete!")
    print()
    
    # Generate and save report
    report = checker.generate_report(results)
    
    # Save report to file (always works with UTF-8)
    report_file = 'factcheck_report.txt'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Try to print to console (may fail on Windows with special characters)
    try:
        print(report)
    except UnicodeEncodeError:
        # Windows console can't handle unicode - just show summary
        print("\n" + "=" * 80)
        print("Report saved to factcheck_report.txt (console cannot display unicode)")
        print("=" * 80)
        # Show just the summary stats
        total = len(results)
        exact = sum(1 for r in results if r.match_type == 'exact')
        claim_not_in_tex = sum(1 for r in results if r.match_type == 'claim_not_in_tex')
        fuzzy = sum(1 for r in results if r.match_type == 'fuzzy')
        wrong = sum(1 for r in results if r.match_type == 'wrong_page')
        not_found = sum(1 for r in results if r.match_type == 'not_found')
        missing = sum(1 for r in results if r.match_type == 'file_missing')
        marked_invalid = sum(1 for r in results if r.match_type == 'marked_invalid')
        print(f"Total: {total} | Exact: {exact} | Claim not in tex: {claim_not_in_tex} | Wrong page: {wrong} | Not found: {not_found} | Missing: {missing} | Marked INVALID: {marked_invalid}")
        print("=" * 80)
    
    print(f"\nReport saved to: {report_file}")
    
    # Return exit code based on results
    # Treat claim_not_in_tex as a FAILURE - CSV must match chapter.tex
    failed = sum(1 for r in results if r.match_type in ['not_found', 'file_missing', 'snippet_too_short', 'invalid_ellipsis', 'claim_not_in_tex'])
    
    if failed > 0:
        print(f"\nFAILED: {failed} citation(s) have errors that must be fixed.")
        if any(r.match_type == 'claim_not_in_tex' for r in results):
            claim_failures = sum(1 for r in results if r.match_type == 'claim_not_in_tex')
            print(f"  WARNING: {claim_failures} citation(s) have claim_text that doesn't exist in chapter.tex")
            print(f"  Your CSV and LaTeX document are OUT OF SYNC!")
    
    return 0 if failed == 0 else 1


def get_hardcoded_citations() -> List[Citation]:
    """
    Fallback function with minimal example citations.
    Only used if claims.csv is not found.
    
    In normal operation, the tool reads from claims.csv.
    """
    print("WARNING: Using fallback hardcoded citations. Please create claims.csv for normal operation.")
    print()
    
    return [
        Citation(
            cite_key='epFactsheet2025',
            pages='p.~1',
            claim_text='Example citation - employment policy has evolved',
            context='Example Section',
            expected_snippet='employment strategy'
        ),
    ]


if __name__ == '__main__':
    exit(main())
