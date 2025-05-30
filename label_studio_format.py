import json
from pathlib import Path
import argparse
import sys
from typing import Iterable
from typing import Union

def text_to_ls_format(
        text_file_path: Path,
        output_folder: Path,
        ref_id: int,
        assigned_to: Union[str, list[str]],   # ← new Union
) -> None:
    output_folder.mkdir(parents=True, exist_ok=True)


    if not text_file_path.exists():
        print(f"file not found: {text_file_path}")
        return

    text_content = text_file_path.read_text(encoding="utf-8")

    doc = {
        "data": {
            "case_content": text_content,
            "ref_id": ref_id,
            "assigned_to": assigned_to,
        }
    }
    out_path = output_folder / f"{text_file_path.stem}.json"
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=4), encoding="utf-8")
    print(f"✓ {assigned_to} ← {text_file_path.name} → {out_path.relative_to(output_folder.parent)}")

    #print(json.dumps(json_structure_content, ensure_ascii=False, indent=4))

# ---------------------------------------------------------------------------
# Extra utility functions and CLI entry-point
# ---------------------------------------------------------------------------

def _collect_txt_files(input_path: Path) -> Iterable[Path]:
    """
    Yield every .txt file that should be converted.

    * If *input_path* is a directory, walk the top level and grab each
      "*.txt" file that lives inside it.
    * If *input_path* is an individual file, check that it ends with
      ".txt" and yield just that one file.
    * Anything else is ignored with a warning.

    Parameters
    ----------
    input_path : Path
        Directory or file supplied on the command line.

    Yields
    ------
    Path
        Path objects pointing at text files ready for conversion.
    """
    if input_path.is_dir():
        yield from input_path.glob("*.txt")      # non-recursive on purpose
    elif input_path.is_file():
        if input_path.suffix.lower() == ".txt":
            yield input_path
        else:
            print(f"[skip] {input_path} is not a .txt file", file=sys.stderr)
    else:
        print(f"[error] Input path does not exist: {input_path}", file=sys.stderr)

def main() -> None:
    """
    Parse command-line arguments and hand work off to *text_to_ls_format*.

    The CLI accepts:

    * **input**  – a single `.txt` file *or* a folder of `.txt` files
    * **output** – destination folder for the generated `.json` files
    * **--start-ref** – optional first reference number (defaults to **1**)

    Example
    -------
    Convert one file::

        python convert_ls.py case01.txt ./json_out

    Convert everything in a folder, numbering refs from 100::

        python convert_ls.py ./cases_txt ./json_out --start-ref 100
    """
    parser = argparse.ArgumentParser(
        description="Convert .txt files into Label-Studio JSON stubs"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .txt file or to a directory that contains .txt files",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Folder where the resulting .json files will be stored",
    )
    parser.add_argument(
        "--start-ref",
        type=int,
        default=1,
        metavar="N",
        help="Reference ID to start counting from (default: 1)",
    )

    args = parser.parse_args()

    for idx, txt_path in enumerate(
            _collect_txt_files(args.input), start=args.start_ref
    ):
        text_to_ls_format(txt_path, args.output, ref_id=idx, assigned_to=args.assigned_to)


if __name__ == "__main__":
    main()