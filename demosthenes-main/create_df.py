import json
import os
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords


def remove_stopwords(text, language):
    stpword = stopwords.words(language)
    # strip punctuation
    no_punctuation = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(w for w in no_punctuation.split() if w.lower() not in stpword)


def uniform(el):
    # sorted list of values for multi-value attributes
    if el == el:  # keep NaN check behavior
        if isinstance(el, list):
            el = sorted(el)
        elif isinstance(el, str) and '|' in el:
            el = sorted(el.split('|'))
    return el


def _filename_id_for_fold(p: Path):
    """
    Extract the numeric id used by crossvalfolds from a file like 'english_1003.json'.
    Falls back gracefully if format differs.
    """
    parts = p.stem.split('_')
    return parts[1] if len(parts) > 1 else parts[0]


def create_df_annotations(path, crossvalfolds=None, *, verbose: bool = False):
    crossvalfolds = crossvalfolds or {}
    in_dir = Path(path)
    files = sorted(p for p in in_dir.glob("*.json") if p.is_file())

    temp = []
    for fp in files:
        with fp.open('r', encoding='utf8') as f:
            data = json.load(f)

        # figure out the split from the filename id (e.g., english_1003.json -> 1003)
        file_key = _filename_id_for_fold(fp)
        split = int(crossvalfolds.get(file_key, 0))

        annotations = data.get("annotations", [])
        doc_rows = []
        for annotation in annotations:
            if annotation.get("name") not in {'conc', 'prem'}:
                continue

            document = data.get("document", {}).get("name", "")
            name = annotation.get("name", "")
            _id = annotation.get("_id", "")
            start = int(annotation.get("start", -1))
            end = int(annotation.get("end", -1))
            plain = data.get("document", {}).get("plainText", "")

            text = plain[start:end] if 0 <= start <= end <= len(plain) else ""

            # attribute extraction, NaN if that attribute is not present
            attrs = annotation.get("attributes", {})
            T = uniform(attrs.get("T", np.nan))
            S = uniform(attrs.get("S", np.nan))

            # clean & trim
            text = clean(text, 'english')
            text = text.lstrip('‘’\'\n0123456789.-–…;;) ')
            text = text.rstrip('‘’\'\n.;; ')

            doc_rows.append([document, split, name, _id, text, T, S])

        temp.extend(doc_rows)

        if verbose:
            counts = Counter(row[2] for row in doc_rows)
            print(
                f"[ANNOTATIONS] file={fp.name} split={split} prem={counts.get('prem', 0)} "
                f"conc={counts.get('conc', 0)}"
            )

    # df creation
    df = pd.DataFrame(
        temp,
        columns=['Document', 'Split', 'Name', 'Id', 'Text', 'Type', 'Scheme']
    )

    # remove stopwords and punctuation from each sentence
    df["Text"] = df["Text"].apply(lambda x: remove_stopwords(x, 'english'))

    # write next to input dir for clarity
    out_path = (Path.cwd() / "df_annotations.pkl")
    df.to_pickle(out_path)


def clean(text, language):
    # deletes ...
    text = re.sub(r'\.\.\.', ' ', text)
    # deletes . from paragraphs' numbers: 2.1 -> 21
    text = re.sub(r'(\d+)\.(\d+)', r'\1\2', text)
    # same if number is 1.2.3. (this overlaps with previous, but keep to mirror original intent)
    text = re.sub(r'(\d+)\.(\d+)\.?', r'\1\2', text)
    # delete . from one-letter words: p., n., A.B.C., ...
    text = re.sub(r'(\w)\.(\w)\.(\w)\.(\w)\.', r'\1\2\3\4', text)
    text = re.sub(r'(\w)\.(\w)\.(\w)\.', r'\1\2\3', text)
    text = re.sub(r'(\w)\.(\w)\.', r'\1\2', text)
    text = re.sub(r'(\W)(\w)\.', r'\1\2', text)
    # some specific abbreviations
    text = re.sub(r'(\W)No\.', r'\1No', text)
    text = re.sub(r'(\W)Dr\.', r'\1Dr', text)
    text = re.sub(r'(\W)seq\.', r'\1seq', text)
    text = text.replace('andamp;', '')
    # delete ; between parenthesis
    for _ in range(6):
        text = re.sub(r'\(([^\)]*);(.*)\)', r'(\1\2)', text)
    # delete \n between : and citation
    text = re.sub(r':(\n)+“(\d)*(\s)*', r': “', text, re.MULTILINE)
    text = re.sub(r':(\n)+‘(\d)*(\s)*', r': ‘', text, re.MULTILINE)
    return text


def create_df_all_sentences(path, crossvalfolds=None, *, verbose: bool = False):
    crossvalfolds = crossvalfolds or {}
    in_dir = Path(path)
    files = sorted(p for p in in_dir.glob("*.json") if p.is_file())

    temp = []
    for fp in files:
        with fp.open('r', encoding='utf8') as f:
            data = json.load(f)

        plainText = data.get("document", {}).get("plainText", "")
        document_name = data.get("document", {}).get("name", fp.stem)

        file_key = _filename_id_for_fold(fp)
        split = int(crossvalfolds.get(file_key, 0))

        # collect raw annotation spans before cleaning so we can check overlaps
        annotation_spans = []
        annotations = data.get("annotations", [])
        doc_rows = []
        counts = Counter()
        invalid_spans = 0
        emptied_spans = 0
        logged_annotated = 0
        logged_void = 0

        # collect annotations first so we can gather stats regardless of sentence matching
        for annotation in annotations:
            if annotation.get("name") not in {'conc', 'prem'}:
                continue

            start = int(annotation.get("start", -1))
            end = int(annotation.get("end", -1))
            if not (0 <= start <= end <= len(plainText)):
                invalid_spans += 1
                continue

            raw_text = plainText[start:end]
            cleaned = clean(raw_text, 'english')
            cleaned = cleaned.lstrip('‘’\'\n\t0123456789.-–…;) ')
            cleaned = cleaned.rstrip('‘’\'\n\t.; ')
            if not cleaned or len(cleaned) <= 5:
                emptied_spans += 1
                continue

            annotation_spans.append((start, end))
            label_name = annotation.get("name", "")
            doc_rows.append([document_name, split, label_name, cleaned])
            counts[label_name] += 1
            if verbose and logged_annotated < 3:
                preview = (cleaned[:80] + '...') if len(cleaned) > 80 else cleaned
                print(f"[ANNOTATED] {document_name} split={split} label={label_name} text={preview}")
                logged_annotated += 1

        # derive sentence candidates (with offsets) from raw plain text
        sentences_with_spans = []
        for match in re.finditer(r'[^.;\n]+', plainText):
            s_start, s_end = match.start(), match.end()
            snippet = match.group()
            cleaned = clean(snippet, 'english')
            cleaned = cleaned.lstrip('‘’\'\n\t0123456789.-–…;) ')
            cleaned = cleaned.rstrip('‘’\'\n\t.; ')
            if not cleaned or len(cleaned) <= 5:
                continue
            sentences_with_spans.append((s_start, s_end, cleaned))

        overlapped = 0

        for s_start, s_end, sentence_text in sentences_with_spans:
            overlaps = any((s_start < ann_end and s_end > ann_start) for ann_start, ann_end in annotation_spans)
            if overlaps:
                overlapped += 1
                continue

            doc_rows.append([document_name, split, 'void', sentence_text])
            counts['void'] += 1
            if verbose and logged_void < 3:
                preview = (sentence_text[:80] + '...') if len(sentence_text) > 80 else sentence_text
                print(
                    f"[VOID] {document_name} split={split} text={preview}"
                )
                logged_void += 1

        temp.extend(doc_rows)

        if verbose:
            print(
                f"[DOCUMENT] {document_name} split={split} prem={counts.get('prem', 0)} "
                f"conc={counts.get('conc', 0)} void={counts.get('void', 0)} overlapped_sentences={overlapped} "
                f"invalid_spans={invalid_spans} empty_after_clean={emptied_spans}"
            )

    # df creation
    df = pd.DataFrame(temp, columns=['Document', 'Split', 'Name', 'Text'])
    # remove stopwords and punctuation from each sentence
    df["Text"] = df["Text"].apply(lambda x: remove_stopwords(x, 'english'))

    out_path = (Path.cwd() / "df_sentences.pkl")
    df.to_pickle(out_path)


# dataframes creation
crossvalfolds = {
    '1000': '1', '1001': '1', '1002': '1', '1003': '2', '1004': '2', '1005': '2',
    '1006': '3', '1007': '3', '1008': '3', '1009': '4', '1010': '4', '1011': '4',
    '1012': '5', '1013': '5', '1014': '5', '1015': '3', '1016': '1', '1017': '1',
    '1018': '1', '1019': '1', '1020': '1', '1021': '2', '1022': '2', '1023': '2',
    '1024': '2', '1025': '5', '1026': '5', '1027': '3', '1028': '3', '1029': '3',
    '1030': '4', '1031': '4', '1032': '4', '1033': '4', '1034': '4', '1035': '2',
    '1036': '3', '1037': '5', '1038': '5', '1039': '5'
}

# Use a POSIX-friendly path (relative or absolute). Example assumes the folder is in CWD:
json_dir = Path("demosthenes_dataset_json")
VERBOSE = os.getenv("DEMOSTHENES_VERBOSE", "0") not in {"0", "false", "False", ""}
create_df_annotations(json_dir, crossvalfolds=crossvalfolds, verbose=VERBOSE)
create_df_all_sentences(json_dir, crossvalfolds=crossvalfolds, verbose=VERBOSE)
