# Annotation fixes

Utilities for repairing Label Studio span offsets in `annotations/final_annotations_iaa_set/`.

## Problem this targets

During annotation, some texts were accidentally modified by inserting or replacing parts of the source text with
strings like:

- `Implicit Intermediate Conclusion [<id>]`

This caused downstream span offsets to drift between annotators (making IAA harder).

## What the fixer does

- Uses the *clean* canonical text from `label_studio_taks/overlapping/*` (`data.case_content`) keyed by `data.ref_id`.
- For each annotation JSON in `annotations/final_annotations_iaa_set/`:
  - Replaces `task.data.case_content` with the clean canonical text.
  - Remaps every non-IIC span `[start,end)` from the (possibly corrupted) text onto the canonical text using a diff
    alignment.
  - Keeps “Implicit Intermediate Conclusion …” spans in `result`, but sets their `start`/`end` to `null` so they can
    still participate in `relation` edges without being treated as real text spans.
- Validates that each remapped non-IIC span matches its stored `value.text` in the canonical text.

## Run

From the repo root:

```bash
python -m annotation_fixes.fix_final_annotations_iaa_set
```

Useful options:

```bash
# Dry-run validation only
python -m annotation_fixes.fix_final_annotations_iaa_set --dry-run

# Choose a custom backup folder
python -m annotation_fixes.fix_final_annotations_iaa_set --backup-dir annotations/final_annotations_iaa_set_backup_custom
```

