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

from .helpers import load_annotation_spans

_DEFAULT_ALLOWED_LABELS: Sequence[str] = (
    "Analysis",
    "Background Facts",
    "Conclusion",
    "Procedural History",
    "Rule",
)


@dataclass
class GPT5Config:
    model: str = "gpt-5.5-mini"
    temperature: float = 1
    max_output_tokens: int = 1024
    max_context_chars: int = 3500
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


_SYSTEM_PROMPT = "You are a precise legal text classifier that must choose exactly one label."

_GUIDELINES_CORE = (
    "- Background Facts: one continuous block (often long); may include IRS administrative actions (audits, assessments), and attributions like 'The Tax Court found...'; must not overlap with other spans; never connect with relations/arrows.\n"
    "- Procedural History: one continuous block; ONLY court procedure (complaints, motions, judgments, appeals, remands, cert petitions/grants). IRS administrative steps belong in Background Facts. Suing in court for a refund is Procedural History; paying IRS/requesting refund/administrative denial are Background Facts. May include recap of lower court’s reasoning. For Supreme Court, include cert petition/grant. Never connect with relations/arrows.\n"
    "- Non-overlap: Background Facts and Procedural History never overlap with each other or with Rule/Analysis/Conclusion.\n"
    "- Ignore early summaries: If the opening 1–3 paragraphs briefly summarize facts/procedure/law/holding and the same material appears later in detail, do NOT annotate those early summaries.\n"
    "- Rule: include legal principles AND their citations (cases, statutes, secondary sources) at the end of the Rule span.\n"
    "- CRAC handling: In a classic CRAC paragraph, initial Conclusion (C) is NOT annotated; Rule (R) is Rule; Analysis (A) is Analysis; the final C is often a second Analysis or a second Rule. Draw arrows/relations from the Rule and first Analysis to that second Analysis. If the conclusion appears only at the start (CRA), label that starting C as Analysis and point R and A back to it.\n"
    "- IRAC/CRAC exclusions: The 'I' in IRAC and the initial 'C' in CRAC are not annotated.\n"
    "- Unannotated exceptions: Court observations (historical/contextual remarks), pure party-argument recitals before evaluation (these are the 'I'), and statements of what the court is not deciding.\n"
    "- Distinguishing precedent: Put precedent’s facts/holding in Rule; explain why it doesn’t apply in Analysis; the resulting non-application conclusion is a second Analysis that points to the surviving Analysis/Conclusion.\n"
)
_CONCLUSION_GUIDELINE = (
    "- Conclusions: At least one Conclusion per case. If only the disposition is stated (e.g., 'Judgment affirmed', 'Summary judgment granted'), annotate that as Conclusion. Multiple issues can yield multiple Conclusions.\n"
)



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
    prompt_header = (
        "Classify the target passage into one of the following labels and following the following guidelines: "
        f"{label_options}. Respond with a single line in the exact format Class:[label].\n\n"
        "Additional Annotation Guidelines:\n"
        f"{guidelines}\n"
    )
    return prompt_header + f"Context:\n{context}\n\nPassage:\n{passage}\n"


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
) -> dict:
    cfg = config or GPT5Config()
    if test_mode:
        cfg.test_mode = True
        cfg.progress = True

    df = load_annotation_spans(annotation_dir)
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
                input=prompt,
                reasoning={"effort": "low"},
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
