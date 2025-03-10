import re

def extract_facts(text: str) -> str:
    
    lines = text.splitlines()
    title_lines = []
    title_start_index = None
    successful_task = False
    
    docket_pattern = re.compile(r'^\s*docket\s+no\.', re.IGNORECASE)
    date_pattern   = re.compile(r'\b(19|20)\d{2}\b')
    
    for i, line in enumerate(lines):
        if re.search(r'united\s+states', line, re.IGNORECASE):
            title_start_index = i
            break
            
    if title_start_index is not None:
        for j in range(title_start_index, len(lines)):
            candidate = lines[j]
            if (not candidate.strip() or        # blank line
                docket_pattern.search(candidate) or
                date_pattern.search(candidate)):
                break
            title_lines.append(candidate)
    
    
    
    ######## Bakcground part
    
    background_lines = []
    start_pattern = re.compile(
    r'^\s*(?:'
    r'(?:\*\d+\s+)?background'  # "*654 BACKGROUND" "BACKGROUND"
    r'|'
    r'findings of fact'        # "FINDINGS OF FACT"
    r')\s*$',
    re.IGNORECASE)
    end_pattern   = re.compile(r'^\s*(?:discussion|opinion)\s*$', re.IGNORECASE)

    background_start_index = None
    background_end_index = None
    
    for i, line in enumerate(lines):
        if start_pattern.match(line):
            background_start_index = i
            break
            
    if background_start_index is not None:
        for j in range(background_start_index + 1, len(lines)):
            if end_pattern.match(lines[j]):
                background_end_index = j
                break

        # If it does not found end, it takes everything until the end
        if background_end_index is None:
            background_end_index = len(lines)
            print("End delimiter (Discussion/Opinion) was not found.")
        else:
            successful_task = True
        
        background_lines = lines[background_start_index+1 : background_end_index]
    else:
        print("Start delimiter (Background/Findings of Fact) was not found.")

    # Combine
    output = []
    if title_lines:
        output.append("TITLE:")
        output.extend(title_lines)
    if background_lines:
        output.append("\nBACKGROUND:")
        output.extend(background_lines)

    return "\n".join(output), successful_task

def extract_provisions(text: str) -> str:
    