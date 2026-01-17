#!/usr/bin/env python3
"""
Check page numbers in claims.csv against citations in chapter.tex

This script:
1. Reads claims.csv to extract expected page numbers for each citation key
2. Parses chapter.tex to find all parencite and cite commands
3. Compares the page numbers used in chapter.tex with those expected in claims.csv
4. Reports mismatches and missing citations
"""

import re
import csv
import difflib
from collections import defaultdict
from pathlib import Path


def parse_claims_csv(csv_path):
    """
    Parse claims.csv and extract citation keys with their expected page numbers.
    Returns a dict: {cite_key: [list of (page_spec, claim_text, context) tuples]}
    """
    claims = defaultdict(list)
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cite_key = row['cite_key'].strip()
            pages = row['pages'].strip()
            claim_text = row.get('claim_text', '').strip()
            context = row.get('context', '').strip()
            claims[cite_key].append((pages, claim_text, context))
    
    return claims


def extract_citations_from_tex(tex_path):
    r"""
    Extract all \parencite and \cite commands from chapter.tex with surrounding context.
    Returns a list of tuples: [(line_num, cite_type, page_spec, cite_key, full_match, context_before), ...]
    """
    citations = []
    
    # Regex to match \parencite[...]{key} or \cite[...]{key}
    citation_pattern = re.compile(
        r'\\(parencite|cite)\s*(?:\[([^\]]*)\])?\s*\{([^}]+)\}'
    )
    
    with open(tex_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    cite_cleanup = re.compile(r'\\(?:paren)?cite\s*(?:\[[^\]]*\])?\s*\{[^}]+\}')

    for idx, line in enumerate(lines):
        line_num = idx + 1
        prev_line = lines[idx - 1] if idx > 0 else ''
        next_line = lines[idx + 1] if idx + 1 < len(lines) else ''

        # Find all citations in this line
        for match in citation_pattern.finditer(line):
            cite_type = match.group(1)  # 'parencite' or 'cite'
            page_spec = match.group(2) if match.group(2) else ''  # Page specification (e.g., 'p.~2')
            cite_keys_raw = match.group(3)  # Citation key(s)
            full_match = match.group(0)
            
            # Capture the sentence portion preceding this citation
            text_before_citation = line[:match.start()]
            text_before_citation = cite_cleanup.sub('', text_before_citation)
            combined = (prev_line + ' ' + text_before_citation).strip()

            if combined:
                sentences = re.split(r'(?<=[.!?])\s+', combined)
                context = sentences[-1]
            else:
                context = combined

            # Capture the sentence immediately following the citation
            text_after_citation = line[match.end():]
            text_after_citation = cite_cleanup.sub('', text_after_citation)
            combined_after = (text_after_citation + ' ' + next_line).strip()

            if combined_after:
                after_sentences = re.split(r'(?<=[.!?])\s+', combined_after)
                context_after = after_sentences[0]
            else:
                context_after = combined_after
            
            # Handle multiple keys separated by comma
            cite_keys = [key.strip() for key in cite_keys_raw.split(',')]
            
            for cite_key in cite_keys:
                citations.append((
                    line_num,
                    cite_type,
                    page_spec.strip(),
                    cite_key,
                    full_match,
                    context,
                    context_after
                ))
    
    return citations


def normalize_page_spec(page_spec):
    """
    Normalize page specifications for comparison.
    Handles variations like 'p.~2', 'pp.~2--3', 'p. 2', etc.
    """
    if not page_spec:
        return ''
    
    # Remove LaTeX escapes and normalize spaces
    normalized = page_spec.replace('~', ' ').replace('\\', '')
    # Remove extra spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized


def find_best_matching_claim(context_snippets, expected_claims):
    """
    Compare each expected claim with multiple context snippets (before/after) and
    return the highest scoring match.
    """

    def normalize(text):
        # Remove citation commands entirely
        text = re.sub(r'\\(?:paren)?cite\s*(?:\[[^\]]*\])?\s*\{[^}]*\}', '', text)
        # Replace other commands with their content if present
        text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
        text = re.sub(r'\\[a-zA-Z]+', ' ', text)
        # Normalise spacing/number formatting
        text = text.replace('{,}', ',').replace('~', ' ')
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    normalised_snippets = []
    for snippet in context_snippets:
        if not snippet:
            continue
        norm = normalize(snippet)
        if norm:
            normalised_snippets.append(norm)

    best_match = None
    best_score = -1.0

    for expected_page, claim_text, section in expected_claims:
        norm_claim = normalize(claim_text)
        if not norm_claim:
            continue

        local_best = -1.0
        for snippet in normalised_snippets:
            score = difflib.SequenceMatcher(None, snippet, norm_claim).ratio()
            local_best = max(local_best, score)

        if local_best > best_score:
            best_score = local_best
            best_match = (expected_page, claim_text, section, local_best)

    return best_match


def compare_citations(claims, citations):
    """
    Compare citations found in chapter.tex with expected claims.
    Returns a report dictionary with mismatches and statistics.
    """
    mismatches = []
    matched = []
    not_in_claims = []
    
    # Group citations by key for analysis
    tex_citations_by_key = defaultdict(list)
    for citation in citations:
        (
            line_num,
            _cite_type,
            page_spec,
            cite_key,
            full_match,
            context_before,
            context_after,
        ) = citation
        tex_citations_by_key[cite_key].append(
            (line_num, page_spec, full_match, context_before, context_after)
        )
    
    # Check each citation in tex against claims
    for cite_key, cite_instances in tex_citations_by_key.items():
        if cite_key not in claims:
            # This citation is not tracked in claims.csv
            for (
                line_num,
                page_spec,
                full_match,
                _context_before,
                _context_after,
            ) in cite_instances:
                not_in_claims.append({
                    'cite_key': cite_key,
                    'line': line_num,
                    'page_in_tex': page_spec,
                    'citation': full_match
                })
        else:
            # Compare page numbers
            expected_claims = claims[cite_key]
            
            for (
                line_num,
                page_spec,
                full_match,
                context_before,
                context_after,
            ) in cite_instances:
                normalized_tex = normalize_page_spec(page_spec)
                
                # Check if this page spec matches any expected page spec
                match_found = False
                
                for expected_page, claim_text, claim_context in expected_claims:
                    normalized_expected = normalize_page_spec(expected_page)
                    if normalized_tex == normalized_expected:
                        match_found = True
                        matched.append({
                            'cite_key': cite_key,
                            'line': line_num,
                            'page': page_spec,
                            'claim_text': claim_text,
                            'context': claim_context
                        })
                        break
                
                if not match_found:
                    # Find the best matching claim based on surrounding text
                    best_match = find_best_matching_claim(
                        [context_before, context_after], expected_claims
                    )
                    
                    mismatches.append({
                        'cite_key': cite_key,
                        'line': line_num,
                        'page_in_tex': page_spec,
                        'expected_claims': expected_claims,
                        'citation': full_match,
                        'best_match': best_match,
                        'total_claims': len(expected_claims)
                    })
    
    # Find claims not cited in tex
    cited_keys = set(tex_citations_by_key.keys())
    claimed_keys = set(claims.keys())
    not_cited = claimed_keys - cited_keys
    
    return {
        'mismatches': mismatches,
        'matched': matched,
        'not_in_claims': not_in_claims,
        'not_cited': not_cited,
        'total_citations': len(citations),
        'unique_keys_in_tex': len(tex_citations_by_key),
        'unique_keys_in_claims': len(claims)
    }


def generate_report(results, output_path=None):
    """
    Generate a human-readable report of the comparison.
    """
    lines = []
    
    lines.append("=" * 80)
    lines.append("PAGE NUMBER VERIFICATION REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary statistics
    lines.append("SUMMARY")
    lines.append("-" * 80)
    lines.append(f"Total citations found in chapter.tex: {results['total_citations']}")
    lines.append(f"Unique citation keys in chapter.tex: {results['unique_keys_in_tex']}")
    lines.append(f"Unique citation keys in claims.csv: {results['unique_keys_in_claims']}")
    lines.append(f"Matched citations: {len(results['matched'])}")
    lines.append(f"Mismatched page numbers: {len(results['mismatches'])}")
    lines.append(f"Citations not tracked in claims.csv: {len(results['not_in_claims'])}")
    lines.append(f"Claims not cited in chapter.tex: {len(results['not_cited'])}")
    lines.append("")
    
    # Page number mismatches (most important)
    if results['mismatches']:
        lines.append("PAGE NUMBER MISMATCHES")
        lines.append("-" * 80)
        lines.append("These citations have different page numbers than expected in claims.csv:")
        lines.append("")
        
        for i, mismatch in enumerate(results['mismatches'], 1):
            lines.append(f"{i}. Line {mismatch['line']}: {mismatch['cite_key']}")
            lines.append(f"   Found in tex: {mismatch['page_in_tex'] or '(no page specified)'}")
            lines.append(f"   Citation: {mismatch['citation']}")
            lines.append("")
            
            # Show the best matching claim
            if mismatch.get('best_match'):
                expected_page, claim_text, context, score = mismatch['best_match']
                lines.append(
                    "   Expected page (claims.csv column 'pages'): "
                    f"{expected_page}"
                )
                lines.append(
                    f"   (Match confidence {score*100:.1f}% based on quoted claim text)"
                )
                lines.append(f"   Section: '{context}'")
                # Truncate long claim text
                claim_preview = claim_text[:150] + "..." if len(claim_text) > 150 else claim_text
                lines.append(f"   Claim: \"{claim_preview}\"")
            else:
                lines.append(f"   Could not determine correct page (source has {mismatch['total_claims']} claims)")
            
            lines.append("")
    else:
        lines.append("✓ No page number mismatches found!")
        lines.append("")
    
    # Citations not in claims
    if results['not_in_claims']:
        lines.append("CITATIONS NOT TRACKED IN CLAIMS.CSV")
        lines.append("-" * 80)
        lines.append("These citations appear in chapter.tex but are not in claims.csv:")
        lines.append("")
        
        # Group by cite_key
        by_key = defaultdict(list)
        for item in results['not_in_claims']:
            by_key[item['cite_key']].append(item)
        
        for cite_key, instances in sorted(by_key.items()):
            lines.append(f"• {cite_key} ({len(instances)} occurrence(s)):")
            for inst in instances[:5]:  # Show first 5 instances
                lines.append(f"    Line {inst['line']}: {inst['page_in_tex'] or '(no page)'}")
            if len(instances) > 5:
                lines.append(f"    ... and {len(instances) - 5} more occurrence(s)")
            lines.append("")
    
    # Claims not cited
    if results['not_cited']:
        lines.append("CLAIMS NOT CITED IN CHAPTER.TEX")
        lines.append("-" * 80)
        lines.append("These citation keys appear in claims.csv but not in chapter.tex:")
        lines.append("")
        
        for cite_key in sorted(results['not_cited']):
            lines.append(f"• {cite_key}")
        lines.append("")
    
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        print(f"Report saved to: {output_path}")
    
    return report_text


def main():
    """Main execution function."""
    # File paths
    base_dir = Path(__file__).parent
    claims_csv = base_dir / 'claims.csv'
    chapter_tex = base_dir / 'test.tex'
    output_report = base_dir / 'page_number_check_report.txt'
    
    # Check files exist
    if not claims_csv.exists():
        print(f"ERROR: claims.csv not found at {claims_csv}")
        return 1
    
    if not chapter_tex.exists():
        print(f"ERROR: chapter.tex not found at {chapter_tex}")
        return 1
    
    print("Parsing claims.csv...")
    claims = parse_claims_csv(claims_csv)
    print(f"  Found {len(claims)} unique citation keys with expected pages")
    
    print("\nParsing chapter.tex...")
    citations = extract_citations_from_tex(chapter_tex)
    print(f"  Found {len(citations)} total citations")
    
    print("\nComparing citations...")
    results = compare_citations(claims, citations)
    
    print("\nGenerating report...")
    report = generate_report(results, output_report)
    
    print("\n" + report)
    
    # Return error code if there are mismatches
    if results['mismatches']:
        return 1
    else:
        return 0


if __name__ == '__main__':
    exit(main())
