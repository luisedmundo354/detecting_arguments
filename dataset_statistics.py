from pathlib import Path
import re
import sys
import statistics

def list_word_cound(file_path):
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as file_in:
        text = file_in.read()
    words = re.findall(r'\b[a-zA-Z]+\b', text)
    return len(words)

def word_count_mean(input_directory):
    input_directory = Path(input_directory)
    counts = []
    for file_path in input_directory.glob('*.txt'):
        count = list_word_cound(file_path)
        print(f'Processing file {file_path.stem}: {count}')
        counts.append(count)
    if counts:
        mean_count = statistics.mean(counts)
        print("Mean word count:", mean_count)
    else:
        print("No text files found.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        word_count_mean(sys.argv[1])
    else:
        word_count_mean('corporate_tax_cases_txt')
