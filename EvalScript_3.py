#!/usr/bin/env python3
"""
EvalScript_3.py — OpenRouter LLM ensemble judge for claim verification

Fixes vs your EvalScript_3:
- Correctly parses "Claim:" lines WITHOUT appending "Expected snippet:" (your root cause).
- Writes predictions CSV continuously (after every claim).
- Resumable: reads JSONL cache + existing predictions CSV and continues.
- Never crashes on empty predictions or identical scores (AUC/ROC becomes "N/A" with a warning).

Example:
export OPENROUTER_API_KEY="YOUR_KEY"

python3 EvalScript_3.py \
  --report factcheck_report_v2.txt \
  --ground_truth claims_with_labels.csv \
  --out_prefix llm_ensemble \
  --models "deepseek/deepseek-chat-v3-0324" "xiaomi/mimo-v2-flash:free" "tngtech/deepseek-r1t2-chimera:free" \
  --k 3 \
  --resume
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Optional deps (script still runs without ROC plot if missing)
try:
    from sklearn.metrics import (
        accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
        classification_report,
    )
except Exception:
    accuracy_score = None
    precision_recall_fscore_support = None
    roc_auc_score = None
    roc_curve = None
    classification_report = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

import requests


# -----------------------------
# Report parsing (CRITICAL FIX)
# -----------------------------
CLAIM_RE = re.compile(r"^\s*Claim:\s*(.+?)\s*$", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"^\s*Evidence:\s*(.*)\s*$", re.IGNORECASE)
EXPECTED_RE = re.compile(r"^\s*Expected\s+(?:snippet|evidence|text)\s*:\s*(.*)\s*$", re.IGNORECASE)
BLANK_RE = re.compile(r"^\s*$")


@dataclass
class ReportRow:
    idx: int
    claim: str
    evidence: str


def parse_factcheck_report(report_path: Path) -> List[ReportRow]:
    """
    Expected block format:
        Claim: ...
        Expected snippet: ...   (ignored)
        Evidence: ...

    FIX: we ONLY take the Claim line content; we do NOT treat other lines as claim continuation.
    """
    rows: List[ReportRow] = []
    claim: Optional[str] = None
    evidence: Optional[str] = None

    with report_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_claim = CLAIM_RE.match(line)
            if m_claim:
                if claim is not None and evidence is not None:
                    rows.append(ReportRow(idx=len(rows), claim=claim.strip(), evidence=evidence.strip()))
                claim = m_claim.group(1).strip()
                evidence = None
                continue

            if EXPECTED_RE.match(line):
                continue

            m_evi = EVIDENCE_RE.match(line)
            if m_evi:
                evidence = m_evi.group(1).strip()
                continue

            # Allow multi-line evidence (rare)
            if evidence is not None and claim is not None and not BLANK_RE.match(line):
                if not (EXPECTED_RE.match(line) or CLAIM_RE.match(line) or EVIDENCE_RE.match(line)):
                    evidence += " " + line.strip()

    if claim is not None and evidence is not None:
        rows.append(ReportRow(idx=len(rows), claim=claim.strip(), evidence=evidence.strip()))

    return rows


# -----------------------------
# Normalization & GT matching
# -----------------------------
PUNCT_RE = re.compile(r"[^a-z0-9\s]+")


def norm_claim(s: str) -> str:
    s = str(s).strip().lower().replace("’", "'")
    s = PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def pick_claim_and_label_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = list(df.columns)

    claim_col = None
    for key in ["claim", "claims", "statement", "text"]:
        for c in cols:
            if key in c.lower():
                claim_col = c
                break
        if claim_col:
            break
    if claim_col is None:
        raise ValueError(f"Could not find a claim column in ground truth CSV. Columns: {cols}")

    label_col = None
    for key in ["label", "truth", "verdict", "ground", "gt"]:
        for c in cols:
            if key in c.lower():
                label_col = c
                break
        if label_col:
            break
    if label_col is None:
        raise ValueError(f"Could not find a label column in ground truth CSV. Columns: {cols}")

    return claim_col, label_col


def label_to_y(label: Any) -> int:
    s = str(label).strip().lower()
    if s in {"true", "1", "supported", "support", "yes"}:
        return 1
    if s in {"false", "0", "refuted", "no", "unsupported", "not_supported", "notsupported"}:
        return 0
    raise ValueError(f"Unrecognized label value: {label!r}")


# -----------------------------
# OpenRouter judge
# -----------------------------
OR_URL = "https://openrouter.ai/api/v1/chat/completions"

SYSTEM_PROMPT = """You are a strict fact-checking judge.
Given a CLAIM and an EVIDENCE SNIPPET, decide if the evidence ENTAILS the claim.

Rules:
- Use ONLY the evidence snippet. Do not use outside knowledge (even if it's common knowledge).
- SUPPORTED only if the evidence clearly supports ALL parts of the claim.
- If any part is missing/unclear, or only "not contradicted", label REFUTED.
- REFUTED covers both contradiction and insufficient evidence.

Return ONLY valid JSON with:
- label: "SUPPORTED" or "REFUTED"
- support_score: number in [0,1] meaning P(claim is supported by evidence)
- rationale: one short sentence
"""

USER_TEMPLATE = """CLAIM:
{claim}

EVIDENCE SNIPPET:
{evidence}

Return JSON only.
"""


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in response.")
    obj = json.loads(m.group(0))
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON is not an object.")
    return obj


def coerce_label(x: Any) -> str:
    s = str(x).strip().upper()
    if s in {"SUPPORTED", "SUPPORT", "ENTAILS", "TRUE", "YES"}:
        return "SUPPORTED"
    return "REFUTED"


def coerce_score(x: Any, label: str) -> float:
    try:
        v = float(x)
    except Exception:
        v = 1.0 if label == "SUPPORTED" else 0.0
    if math.isnan(v) or math.isinf(v):
        v = 1.0 if label == "SUPPORTED" else 0.0
    return max(0.0, min(1.0, v))


def openrouter_chat(model: str, api_key: str, claim: str, evidence: str,
                    timeout_s: int = 60, max_retries: int = 6, sleep_s: float = 0.25) -> Tuple[Dict[str, Any], str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "http://localhost",
        "X-Title": "ThesisFactCheckEnsemble",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(claim=claim, evidence=evidence)},
        ],
    }

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(OR_URL, headers=headers, json=payload, timeout=timeout_s)
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            obj = extract_json_object(content)
            return obj, content
        except Exception as e:
            last_err = e
            backoff = sleep_s * (2 ** (attempt - 1)) + random.random() * 0.25
            time.sleep(min(backoff, 15.0))

    raise RuntimeError(f"OpenRouter call failed after {max_retries} retries: {last_err}")


# -----------------------------
# Cache + CSV writing
# -----------------------------
def load_jsonl_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ck = obj.get("cache_key")
                if ck:
                    cache[ck] = obj
            except Exception:
                continue
    return cache


def append_jsonl(cache_path: Path, obj: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()


def atomic_write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def model_to_colprefix(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model).strip("_").lower()


def ensure_predictions_schema(df: pd.DataFrame, models: List[str]) -> pd.DataFrame:
    base_cols = ["claim_norm", "claim", "evidence", "gt_label", "gt_y", "ensemble_score", "ensemble_label"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = pd.NA
    for m in models:
        pref = model_to_colprefix(m)
        for c in [f"{pref}_label", f"{pref}_score"]:
            if c not in df.columns:
                df[c] = pd.NA
    return df


def compute_ensemble(row: pd.Series, models: List[str]) -> Tuple[float, str]:
    scores: List[float] = []
    labels: List[str] = []
    for m in models:
        pref = model_to_colprefix(m)
        lab = row.get(f"{pref}_label", None)
        sc = row.get(f"{pref}_score", None)
        if pd.isna(lab) or pd.isna(sc):
            continue
        labels.append(str(lab).upper())
        try:
            scores.append(float(sc))
        except Exception:
            continue

    if not scores:
        return float("nan"), "REFUTED"

    mean_score = sum(scores) / len(scores)
    sup = sum(1 for l in labels if l == "SUPPORTED")
    ref = len(labels) - sup

    if sup > ref:
        ens_label = "SUPPORTED"
    elif ref > sup:
        ens_label = "REFUTED"
    else:
        # tie-break (conservative)
        ens_label = "SUPPORTED" if mean_score >= 0.5 else "REFUTED"

    return mean_score, ens_label


def finalize_metrics(pred_df: pd.DataFrame, out_prefix: str, models: List[str]) -> None:
    usable = pred_df.dropna(subset=["gt_y", "ensemble_score"], how="any").copy()
    if usable.empty:
        print("WARNING: No predictions with ground truth to finalize.")
        return

    y_true = usable["gt_y"].astype(int).tolist()
    y_score = usable["ensemble_score"].astype(float).tolist()
    y_pred = [1 if s >= 0.5 else 0 for s in y_score]

    metrics_path = Path(f"{out_prefix}_metrics.csv")
    report_path = Path(f"{out_prefix}_classification_report.txt")
    roc_path = Path(f"{out_prefix}_roc.png")

    rows = [{"metric": "n", "value": len(y_true)}, {"metric": "models", "value": " | ".join(models)}]

    if accuracy_score is None:
        print("WARNING: sklearn not installed; skipping metrics/ROC.")
        return

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    rows += [
        {"metric": "accuracy", "value": acc},
        {"metric": "precision", "value": p},
        {"metric": "recall", "value": r},
        {"metric": "f1", "value": f1},
    ]

    auc = None
    try:
        if len(set(y_true)) < 2:
            raise ValueError("Only one class in y_true; AUC undefined.")
        if len(set(y_score)) < 2:
            raise ValueError("All scores identical; AUC undefined.")
        auc = roc_auc_score(y_true, y_score)
        rows.append({"metric": "roc_auc", "value": auc})
    except Exception as e:
        rows.append({"metric": "roc_auc", "value": "NaN"})
        print(f"WARNING: ROC-AUC not computed: {e}")

    pd.DataFrame(rows).to_csv(metrics_path, index=False)
    print(f"Wrote metrics: {metrics_path}")

    if classification_report is not None:
        rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
        report_path.write_text(rep, encoding="utf-8")
        print(f"Wrote classification report: {report_path}")

    if plt is not None and roc_curve is not None and auc is not None:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure()
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (LLM Ensemble)")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(roc_path, dpi=200)
        plt.close()
        print(f"Wrote ROC curve: {roc_path}")
    else:
        if plt is None:
            print("NOTE: matplotlib not available; ROC plot skipped.")
        elif auc is None:
            print("NOTE: AUC undefined; ROC plot skipped.")


# -----------------------------
# CLI + main
# -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--ground_truth", required=True)
    p.add_argument("--out_prefix", default="llm_ensemble")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--k", type=int, default=None, help="Use first K models from --models (default: all)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--match_by_order_if_needed", action="store_true",
                   help="If 0 matches, align by row order (ONLY if both files are aligned).")
    p.add_argument("--max_claims", type=int, default=None)
    p.add_argument("--sleep_s", type=float, default=0.25)
    p.add_argument("--timeout_s", type=int, default=60)
    return p


def main() -> None:
    args = build_argparser().parse_args()

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_TOKEN")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY in your environment.")
        sys.exit(2)

    report_path = Path(args.report)
    gt_path = Path(args.ground_truth)
    out_prefix = args.out_prefix

    models = args.models[: args.k] if args.k else args.models

    cache_path = Path(f"{out_prefix}_cache.jsonl")
    pred_path = Path(f"{out_prefix}_predictions.csv")

    report_rows = parse_factcheck_report(report_path)
    if args.max_claims:
        report_rows = report_rows[: args.max_claims]
    print(f"Parsed report rows: {len(report_rows)}")

    gt_df = pd.read_csv(gt_path)
    claim_col, label_col = pick_claim_and_label_columns(gt_df)
    gt_df = gt_df[[claim_col, label_col]].copy()
    gt_df.rename(columns={claim_col: "claim", label_col: "gt_label"}, inplace=True)
    gt_df["claim_norm"] = gt_df["claim"].map(norm_claim)
    gt_df["gt_y"] = gt_df["gt_label"].map(label_to_y)

    rep_df = pd.DataFrame([{"idx": r.idx, "claim": r.claim, "evidence": r.evidence} for r in report_rows])
    rep_df["claim_norm"] = rep_df["claim"].map(norm_claim)

    merged = rep_df.merge(gt_df[["claim_norm", "gt_label", "gt_y"]], on="claim_norm", how="left", indicator=True)
    matched = merged[merged["_merge"] == "both"].copy()

    if matched.empty:
        print("ERROR: 0 matches between report and ground truth.")
        if args.match_by_order_if_needed:
            print("Falling back to match-by-order (ONLY if files aligned).")
            n = min(len(rep_df), len(gt_df))
            matched = rep_df.iloc[:n].copy()
            matched["gt_label"] = gt_df["gt_label"].iloc[:n].values
            matched["gt_y"] = gt_df["gt_y"].iloc[:n].values
        else:
            print("Try: --match_by_order_if_needed (only if both files have same row order)")
            return

    print(f"Matched {len(matched)} claims with ground truth.")

    cache = load_jsonl_cache(cache_path) if args.resume else {}
    if args.resume:
        print(f"Loaded cache entries: {len(cache)}")

    if args.resume and pred_path.exists():
        pred_df = pd.read_csv(pred_path)
    else:
        pred_df = pd.DataFrame()
    pred_df = ensure_predictions_schema(pred_df, models).set_index("claim_norm", drop=False)

    try:
        for n_done, row in enumerate(matched.reset_index(drop=True).itertuples(index=False), start=1):
            claim = str(row.claim)
            evidence = str(row.evidence)
            claim_norm = str(row.claim_norm)
            gt_label = row.gt_label
            gt_y = int(row.gt_y)

            if claim_norm not in pred_df.index:
                pred_df.loc[claim_norm, "claim_norm"] = claim_norm
                pred_df.loc[claim_norm, "claim"] = claim
                pred_df.loc[claim_norm, "evidence"] = evidence
                pred_df.loc[claim_norm, "gt_label"] = gt_label
                pred_df.loc[claim_norm, "gt_y"] = gt_y

            for model in models:
                pref = model_to_colprefix(model)
                if not pd.isna(pred_df.loc[claim_norm].get(f"{pref}_label", pd.NA)) and not pd.isna(
                    pred_df.loc[claim_norm].get(f"{pref}_score", pd.NA)
                ):
                    continue

                ck = sha256_hex(model + "|" + claim_norm + "|" + norm_claim(evidence))
                if ck in cache:
                    pred_df.loc[claim_norm, f"{pref}_label"] = cache[ck].get("label", "REFUTED")
                    pred_df.loc[claim_norm, f"{pref}_score"] = cache[ck].get("support_score", 0.0)
                    continue

                obj, raw = openrouter_chat(
                    model=model,
                    api_key=api_key,
                    claim=claim,
                    evidence=evidence,
                    timeout_s=args.timeout_s,
                    sleep_s=max(args.sleep_s, 0.05),
                )

                label = coerce_label(obj.get("label", "REFUTED"))
                score = coerce_score(obj.get("support_score", 1.0 if label == "SUPPORTED" else 0.0), label)

                rec = {
                    "cache_key": ck,
                    "model": model,
                    "claim": claim,
                    "claim_norm": claim_norm,
                    "evidence": evidence,
                    "label": label,
                    "support_score": score,
                    "rationale": str(obj.get("rationale", "")).strip(),
                    "raw": raw,
                    "ts": time.time(),
                }
                append_jsonl(cache_path, rec)
                cache[ck] = rec

                pred_df.loc[claim_norm, f"{pref}_label"] = label
                pred_df.loc[claim_norm, f"{pref}_score"] = score

            ens_score, ens_label = compute_ensemble(pred_df.loc[claim_norm], models)
            pred_df.loc[claim_norm, "ensemble_score"] = ens_score
            pred_df.loc[claim_norm, "ensemble_label"] = ens_label

            atomic_write_csv(pred_path, pred_df.reset_index(drop=True))
            print(f"[{n_done}/{len(matched)}] wrote: {pred_path}")

    except KeyboardInterrupt:
        print("\nInterrupted. Saving predictions and exiting. Resume with --resume.")
        atomic_write_csv(pred_path, pred_df.reset_index(drop=True))
        return

    atomic_write_csv(pred_path, pred_df.reset_index(drop=True))
    print(f"Final predictions saved: {pred_path}")
    finalize_metrics(pred_df.reset_index(drop=True), out_prefix, models)


if __name__ == "__main__":
    main()
