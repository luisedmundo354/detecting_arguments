from pathlib import Path
from striprtf.striprtf import rtf_to_text
import os

def process_file(file_path):
    """
    Function to process a file.
    Replace this docstring with your actual processing code.
    """
    if not file_path.exists():
        print("File not found:", rtf_file_path.resolve())
    else:
        with file_path.open('r') as file:
            rtf_content = file.read()

        plain_text = rtf_to_text(rtf_content)
        # Perform your processing here
        print(f"Processed {file_path}")
        return plain_text
    
if __name__ == "__main__":

    # Define the directory containing the files
    directory = Path('revised_cases')
    output_directory = Path('precedents')

    # Iterate over all .txt files in the directory
    for file_path in directory.glob('*.rtf'):
        plain_text = process_file(file_path)
        
        output_file_path = output_directory / f"{file_path.stem}.txt"
        
        with open(output_file_path, "w", encoding="utf-8") as file:
            file.write(plain_text)

            print(f"Document {file_path.stem} has been saved")
        
