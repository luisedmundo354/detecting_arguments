import os
import re
import json
from pathlib import Path
import xml.etree.ElementTree as ET


# Creates the <body> tag, that encapsulates the whole file
def insert_body_tag(filepath: Path):
    with filepath.open('r+', encoding='utf8') as f:
        content = f.read()
        if not content.split('>')[0] == '<body':
            f.seek(0, 0)
            f.write('<body>\n' + content.replace('&', 'and') + '\n</body>')


# Makes the first letter of the name upper case
def make_first_cap(name):
    return name[0].upper() + name[1:]


# Returns the list of attributes taking as input the values written as 'Prem1|Prem2'
def make_value_list(values):
    return [val for val in values.split('|')]


# Checks if an XML element has sub-elements
def has_children(element):
    return len(list(element)) > 0


# Main conversion from XML to JSON
# plain_text_presence is a boolean variable that indicates whether we want the plain text in the JSON file or not
def convert_to_json(xml_files_path, base_id=0, plain_text_presence=True, language='english', change_name=True):
    xml_dir = Path(xml_files_path)
    out_dir = Path('./demosthenes_dataset_json')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Only iterate XML files
    xml_files = sorted([p for p in xml_dir.iterdir() if p.is_file() and p.suffix.lower() == '.xml'])

    count = 0
    for xml_path in xml_files:
        print(xml_path.name)

        # Inserting the tag <body> that encapsulates the whole file
        insert_body_tag(xml_path)

        # DOCUMENT
        # Name
        file_name = xml_path.stem

        # ID
        file_id = str(base_id + count)
        print(f"{count}\t{file_id}")
        count += 1

        tree = ET.parse(str(xml_path))
        root = tree.getroot()

        # Plain Text
        plain_text = ''.join(root.itertext())
        plain_text = re.sub(r'\\([^rnt])', r'/\1', plain_text)

        # ANNOTATIONS
        annotations = []
        for child in root.iter():
            if child.tag != 'body':
                tag = {}
                # Document
                tag['document'] = file_id
                # Name
                tag['name'] = child.tag
                # _id
                if 'ID' in child.attrib:
                    tag['_id'] = child.attrib['ID']

                # Attributes
                attributes = {}
                for attr, val in child.attrib.items():
                    if '|' in val and val != 'L|F':
                        attributes[attr] = make_value_list(val)
                    else:
                        attributes[attr] = val
                tag['attributes'] = attributes

                # Position
                tag_text = ''.join(child.itertext())
                tag['start'] = plain_text.find(tag_text)
                tag['end'] = tag['start'] + len(tag_text) if tag['start'] != -1 else -1

                annotations.append(tag)

        # Creating the internal file name
        internal_file_name = f"{language}_{file_id}" if change_name else file_name

        # Writing the JSON file (fixes bad path and quoting)
        json_path = out_dir / f"{internal_file_name}.json"

        # Build the JSON object instead of manual string concatenation
        doc_obj = {
            "document": {
                "_id": file_id,
                "name": file_name
            },
            "annotations": annotations
        }
        if plain_text_presence:
            # escape control characters but keep JSON valid
            doc_obj["document"]["plainText"] = (
                plain_text.replace('"', "'")
                .replace('\n', '\\n')
                .replace('\t', '\\t')
                .replace('\r', '\\r')
            )

        with json_path.open('w', encoding='utf8') as jf:
            json.dump(doc_obj, jf, ensure_ascii=False)

    print(f"Done. Wrote {count} files to {out_dir.resolve()}")


if __name__ == "__main__":
    convert_to_json('demosthenes_dataset', base_id=1000, plain_text_presence=True, language='english')
