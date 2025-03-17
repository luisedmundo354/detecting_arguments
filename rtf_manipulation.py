from pathlib import Path
from striprtf.striprtf import rtf_to_text

def process_file(file_path: Path) -> str:
    """
    Process a single RTF file and return its plain text.
    """
    if not file_path.exists():
        print("File not found:", file_path.resolve())
        return ""
    
    with file_path.open('r', encoding="utf-8") as file:
        rtf_content = file.read()

    plain_text = rtf_to_text(rtf_content)
    print(f"Processed {file_path.name}")
    return plain_text

def process_directory(input_directory: str, output_directory: str):
    """
    Process all .rtf files in the input_directory by converting them to plain text
    and saving the results in the output_directory.
    """
    in_dir = Path(input_directory)
    out_dir = Path(output_directory)
    
    # Create the output directory if it doesn't exist
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Iterate over all .rtf files in the input directory
    for file_path in in_dir.glob('*.rtf'):
        plain_text = process_file(file_path)
        
        # Write the processed text to a new .txt file in the output directory
        output_file_path = out_dir / f"{file_path.stem}.txt"
        with output_file_path.open("w", encoding="utf-8") as out_file:
            out_file.write(plain_text)
            print(f"Document {file_path.stem} has been saved to {output_file_path}")
