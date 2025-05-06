import json
from pathlib import Path

def text_to_ls_format(text_file_path: Path, output_folder: Path, ref_id: int):

    output_folder.mkdir(parents=True, exist_ok=True)

    if not text_file_path.exists():
        print(f"file not found: {text_file_path}")
        return

    text_content = text_file_path.read_text(encoding="utf-8")

    json_structure_content = {
        "data": {
            "case_content": text_content,
            "ref_id": ref_id,
        }
    }

    output_path = output_folder / f"{text_file_path.stem}.json"

    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(json_structure_content, json_file, ensure_ascii=False, indent=4)

    print(f"Document saved to: {output_path}")
    #print(json.dumps(json_structure_content, ensure_ascii=False, indent=4))