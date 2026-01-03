from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report
from tqdm import tqdm

from .helpers import (
    filter_implicit_conclusions as filter_implicit_conclusions_df,
    load_annotation_contents,
    load_annotation_spans,
)

_DEFAULT_ALLOWED_LABELS: Sequence[str] = (
    "Analysis",
    "Background Facts",
    "Conclusion",
    "Procedural History",
    "Rule",
)


@dataclass
class GPT5Config:
    model: str = "gpt-5-mini"
    temperature: float = 1
    max_output_tokens: int = 100000
    max_context_chars: int = 250000
    progress: bool = True
    test_mode: bool = False
    test_limit: int = 10


def _default_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


_CLASS_PATTERN = re.compile(r"class\s*:\s*(?P<label>[A-Za-z\s]+)", re.IGNORECASE)


def _parse_label(text: str, allowed_labels: Sequence[str]) -> Optional[str]:
    if not text:
        return None
    match = _CLASS_PATTERN.search(text)
    if not match:
        return None
    label = match.group("label").strip()
    for candidate in allowed_labels:
        if candidate.lower() == label.lower():
            return candidate
    return None


_SYSTEM_PROMPT = (
    "You are a precise legal passage classifier for U.S. judicial opinions. "
    "Choose exactly ONE label from the allowed set and output ONLY one line in the format "
    "Class:<label>. No extra words, no punctuation, no explanation."
)

_GUIDELINES_CORE = (
    "Annotation scheme (match human annotators): spans are labeled by FUNCTIONAL ROLE in a "
    "chain of syllogisms (polysyllogistic IRAC). Rules and Analyses are argument nodes; "
    "Background Facts and Procedural History are contextual blocks and are not part of the reasoning chain.\n\n"
    "LABEL DEFINITIONS (use these meanings):\n"
    "- Background Facts: narrative context about what happened outside the courtroom (events, transactions, parties, dates). "
    "Includes IRS/agency administrative steps (audits, assessments, refund claims/denials). "
    "These spans inform the reader but do not themselves apply a rule or draw an inference toward the holding.\n"
    "- Procedural History: court/litigation process and posture (complaints, motions, hearings, judgments, appeals, remands, "
    "petitions for certiorari/grants). Focus is the procedural timeline in court. "
    "IRS administrative steps are NOT procedural history.\n"
    "- Rule: a generally applicable premise used to justify an inference—statutes, regulations, precedent holdings, tests, "
    "definitions, and other reusable generalizations (including implicit/brute premises that license an inference). "
    "Summaries of precedent (facts/holdings) used as authority count as Rule. Citations often appear but are not required.\n"
    "- Analysis: case-specific reasoning that applies/interprets a Rule using this case’s facts/record; evaluates evidence; "
    "accepts/rejects/distinguishes arguments; draws causal/logical inferences. "
    "IMPORTANT: intermediate/local conclusions in a reasoning chain are labeled Analysis (even if phrased 'we conclude...' "
    "or 'therefore...') when they support a later step.\n\n"
    "TIE-BREAKERS:\n"
    "1) Court procedure => Procedural History. IRS/admin steps => Background Facts.\n"
    "2) Stating a legal standard / definition / precedent holding => Rule. Applying or distinguishing it here => Analysis.\n"
    "3) If both appear, choose the dominant function: 'state the law/test' => Rule; 'apply to facts / infer' => Analysis.\n"
)
_CONCLUSION_GUIDELINE = (
    "CONCLUSION (only when this label exists in the allowed set):\n"
    "- Conclusion: ONLY the terminal outcome of an argument tree / issue—i.e., the court’s ultimate holding or disposition "
    "(e.g., 'judgment affirmed/reversed', 'summary judgment granted', 'the deduction is allowed/denied'). "
    "Do NOT use Conclusion for intermediate steps that can function as premises for later reasoning; those are Analysis (or Rule).\n"
)



def _normalize_ws(text: str) -> str:
    """Collapse whitespace to improve passage matching and make excerpts stable."""

    return " ".join((text or "").split())


def _build_case_context(case_text: str, passage: str, max_chars: int) -> str:
    """
    Build a length-capped case excerpt that helps with:
      - local classification (window around passage)
      - detecting terminal dispositions (end excerpt)
      - procedural posture (begin excerpt)
    """

    case_norm = _normalize_ws(case_text)
    passage_norm = _normalize_ws(passage)
    if not case_norm:
        return ""
    if len(case_norm) <= max_chars:
        return case_norm

    # Allocate fixed budgets so we always include end-of-case signals.
    head_len = max(300, int(max_chars * 0.22))
    tail_len = max(300, int(max_chars * 0.22))
    local_len = max(300, max_chars - head_len - tail_len - 200)  # ~200 chars for headers

    head = case_norm[:head_len]
    tail = case_norm[-tail_len:]

    idx = case_norm.find(passage_norm)
    if idx == -1 and passage_norm:
        # Try a smaller anchor (first ~20 tokens) if the full passage doesn't match exactly.
        anchor = " ".join(passage_norm.split()[:20])
        idx = case_norm.find(anchor) if anchor else -1

    if idx != -1:
        half = local_len // 2
        start = max(0, idx - half)
        end = min(len(case_norm), idx + len(passage_norm) + half)
        local = case_norm[start:end]
    else:
        # Fallback: mid excerpt
        mid_start = max(0, (len(case_norm) // 2) - (local_len // 2))
        local = case_norm[mid_start : mid_start + local_len]

    excerpt = (
        "NOTE: Case text is excerpted due to length.\n"
        "=== BEGINNING OF CASE (excerpt) ===\n"
        f"{head}\n\n"
        "=== LOCAL CONTEXT (excerpt) ===\n"
        f"{local}\n\n"
        "=== END OF CASE (excerpt) ===\n"
        f"{tail}\n"
    )
    return excerpt[:max_chars]


def _build_context(texts: Iterable[str], max_chars: int) -> str:
    buffer: list[str] = []
    total = 0
    for text in texts:
        if not text:
            continue
        remaining = max_chars - total
        if remaining <= 0:
            break
        snippet = text[:remaining]
        buffer.append(snippet)
        total += len(snippet) + 1
    return "\n".join(buffer)


def _merge_conclusion_into_analysis(value: object) -> object:
    if isinstance(value, str) and value.strip().lower() == "conclusion":
        return "Analysis"
    return value


def _get_allowed_labels(combine: bool) -> list[str]:
    labels = list(_DEFAULT_ALLOWED_LABELS)
    if combine:
        labels = [label for label in labels if label != "Conclusion"]
    return labels


def _format_label_options(labels: Sequence[str]) -> str:
    if not labels:
        raise ValueError("At least one label must be provided.")
    if len(labels) == 1:
        return labels[0]
    return ", ".join(labels[:-1]) + f", or {labels[-1]}"


def _render_prompt(
    *,
    context: str,
    passage: str,
    allowed_labels: Sequence[str],
    include_conclusion_guidance: bool,
) -> str:
    guidelines = _GUIDELINES_CORE
    if include_conclusion_guidance:
        guidelines += _CONCLUSION_GUIDELINE
    label_options = _format_label_options(list(allowed_labels))
    has_conclusion = any(lbl.lower() == "conclusion" for lbl in allowed_labels)
    conclusion_fallback = ""
    if not has_conclusion:
        conclusion_fallback = (
            "IMPORTANT: 'Conclusion' is NOT an available label in this run. "
            "If the passage states the final outcome/disposition, label it as Analysis.\n\n"
        )

    prompt = (
        "TASK\n"
        "You will be given (1) case text (context) and (2) one TARGET PASSAGE from that case.\n"
        f"Choose exactly ONE label from: {label_options}\n\n"
        "OUTPUT FORMAT (STRICT)\n"
        "Return exactly ONE line:\n"
        "Class:<label>\n"
        "Do not output anything else.\n\n"
        f"{conclusion_fallback}"
        "GUIDELINES (match the human annotators)\n"
        f"{guidelines}\n\n"
        "<<<CASE_TEXT>>>\n"
        f"{context}\n"
        "<<<END_CASE_TEXT>>>\n\n"
        "<<<TARGET_PASSAGE>>>\n"
        f"{passage}\n"
        "<<<END_TARGET_PASSAGE>>>\n"
    )
    return prompt


def _resolve_checkpoint_path(
    checkpoint_path: Optional[Path | str],
    *,
    combine_analysis_conclusion: bool,
) -> Optional[Path]:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path)
    if combine_analysis_conclusion:
        suffix = path.suffix
        combined_name = f"{path.stem}_analysis_merged{suffix}"
        path = path.with_name(combined_name)
    return path


def run_gpt5_classification(
    annotation_dir,
    *,
    client: Optional[OpenAI] = None,
    config: Optional[GPT5Config] = None,
    test_mode: bool = False,
    checkpoint_path: Optional[Path | str] = None,
    combine_analysis_conclusion: bool = False,
    filter_implicit_conclusions: bool = False,
) -> dict:
    cfg = config or GPT5Config()
    if test_mode:
        cfg.test_mode = True
        cfg.progress = True

    df = load_annotation_spans(annotation_dir)
    if filter_implicit_conclusions:
        df = filter_implicit_conclusions_df(df)
    contents_by_doc = load_annotation_contents(annotation_dir)
    if combine_analysis_conclusion:
        df = df.copy()
        df["label"] = df["label"].apply(_merge_conclusion_into_analysis)
    if client is None:
        client = _default_client()

    allowed_labels = _get_allowed_labels(combine_analysis_conclusion)
    labels = df["label"].to_numpy()
    doc_entries = {
        doc: list(zip(group.index.tolist(), group["text"].tolist()))
        for doc, group in df.groupby("document_id", sort=False)
    }

    checkpoint_file: Optional[Path] = None
    existing: Dict[str, str] = {}
    writer = None
    resolved_checkpoint = _resolve_checkpoint_path(
        checkpoint_path,
        combine_analysis_conclusion=combine_analysis_conclusion,
    )
    if not cfg.test_mode and resolved_checkpoint is not None:
        checkpoint_file = resolved_checkpoint
        if checkpoint_file.exists():
            with checkpoint_file.open("r", encoding="utf8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    span_id = record.get("span_id")
                    pred = record.get("prediction")
                    if span_id and pred:
                        if combine_analysis_conclusion:
                            pred = _merge_conclusion_into_analysis(pred)
                        existing[span_id] = pred
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        writer = checkpoint_file.open("a", encoding="utf8")

    span_ids = df["span_id"].tolist()
    predictions_buffer = [existing.get(span_id) for span_id in span_ids]

    total_rows = len(df)
    limit = cfg.test_limit if cfg.test_mode else total_rows
    available_positions = [idx for idx in range(total_rows) if predictions_buffer[idx] is None]
    if cfg.test_mode:
        available_positions = available_positions[:limit]

    iterator = (
        tqdm(available_positions, total=len(available_positions), desc="gpt-5-mini")
        if cfg.progress
        else available_positions
    )

    if cfg.test_mode:
        print(f"[test_mode] Limiting GPT queries to {len(available_positions)} samples")

    processed_positions: list[int] = []
    try:
        for pos in iterator:
            row_index = df.index[pos]
            span_id = span_ids[pos]

            text = df.at[row_index, "text"]
            doc = df.at[row_index, "document_id"]
            entries = doc_entries.get(doc, [])
            case_text = contents_by_doc.get(doc, "") if contents_by_doc else ""
            context = _build_case_context(case_text, text, cfg.max_context_chars)
            if not context:
                other_texts = (t for j, t in entries if j != row_index)
                context = _build_context(other_texts, cfg.max_context_chars)
            prompt = _render_prompt(
                context=context or "No additional context.",
                passage=text,
                allowed_labels=allowed_labels,
                include_conclusion_guidance=not combine_analysis_conclusion,
            )
            if cfg.test_mode:
                print("\n[test_mode] Prompt:\n" + prompt)

            response = client.responses.create(
                model=cfg.model,
                instructions=_SYSTEM_PROMPT,
                input=prompt,
                reasoning={"effort": "high"},
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
            )
            print("This is the response:", response)
            raw_text = getattr(response, "output_text", "") or ""
            if cfg.test_mode:
                print("[test_mode] Raw response:\n" + raw_text)
            label = _parse_label(raw_text, allowed_labels)
            if cfg.test_mode:
                print(f"[test_mode] Parsed label: {label or 'Unknown'}")
            final_label = label or "Unknown"
            if combine_analysis_conclusion:
                final_label = _merge_conclusion_into_analysis(final_label)
            predictions_buffer[pos] = final_label
            processed_positions.append(pos)

            if writer is not None:
                writer.write(json.dumps({"span_id": span_id, "prediction": final_label}) + "\n")
                writer.flush()
    finally:
        if writer is not None:
            writer.close()

    if cfg.test_mode:
        positions = processed_positions
    else:
        positions = list(range(total_rows))
        missing = [i for i, value in enumerate(predictions_buffer) if value is None]
        if missing:
            raise RuntimeError(
                f"Missing predictions for {len(missing)} spans; checkpoint may be incomplete."
            )

    output_predictions = np.array([predictions_buffer[pos] for pos in positions])
    output_labels = labels[positions]

    report = classification_report(
        output_labels,
        output_predictions,
        labels=allowed_labels,
        target_names=allowed_labels,
        zero_division=0,
        output_dict=True,
    )

    out_df = df.iloc[positions].copy()
    out_df["prediction_gpt5"] = output_predictions

    return {
        "report": report,
        "predictions": out_df[["span_id", "label", "prediction_gpt5"]],
        "processed_rows": len(positions),
        "checkpoint_path": str(checkpoint_file) if checkpoint_file else None,
    }
