import json
import argparse
import span_iaa_ned as iaa_ned
import krippendorff

def load_annotations_from_file(path):
    """Load a single json"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Identify annotator
    completed_by = data.get('completed_by', {}) or {}
    annotator = completed_by.get('email') or data.get('created_username') or path
    # Extract spans per category
    ann = {}
    labels_set = set()
    for item in data.get('result', []):
        if item.get('type') != 'labels':
            continue
        cats = item.get('value', {}).get('labels', [])
        text = item.get('value', {}).get('text', '').strip()
        for cat in cats:
            labels_set.add(cat)
            ann.setdefault(cat, []).append(text)
    return annotator, ann, labels_set
    
def compute_krippendorff_alpha(all_annotations, categories):
    alpha_by_cat = {}
    annotators = sorted(all_annotations.keys())
    for cat in categories:
        # Collect all unique spans for this category
        units = sorted({u for ann in all_annotations.values() for u in ann.get(cat, [])})
        if not units:
            alpha_by_cat[cat] = 1.0
            continue
        # Build reliability data: rows = annotators, cols = units
        reliability_data = []
        for a in annotators:
            spans = set(all_annotations.get(a, {}).get(cat, []))
            reliability_data.append([1 if u in spans else 0 for u in units])
        # Compute alpha
        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement='nominal'
        )
        alpha_by_cat[cat] = alpha
    return alpha_by_cat

def main():
    parser = argparse.ArgumentParser(
        description='Compute IAA from Label Studio JSON files.'
    )
    parser.add_argument(
        'json_files', nargs='+',
        help='Paths to Label Studio JSON export files (one annotator per file).'
    )
    args = parser.parse_args()

    if len(args.json_files) < 2:
        parser.error('Need at least two JSON files to compute inter-annotator agreement.')

    all_annotations = {}
    categories_set = set()
    for path in args.json_files:
        annotator, ann, labels = load_annotations_from_file(path)
        all_annotations[annotator] = ann
        categories_set.update(labels)

    categories = sorted(categories_set)
    # Compute IAA (micro-average pairwise F1 per category)
    f1_by_cat = iaa_ned.compute_iaa(all_annotations, categories)
    # Compute Krippendorff's alpha for each category
    alpha_by_cat = compute_krippendorff_alpha(all_annotations, categories)

    # Output results
    print('Categories:', categories)
    print('IAA F1 by category:')
    for cat in categories:
        score = f1_by_cat.get(cat, 0.0)
        print(cat, ': ', score)
    # Output Krippendorff's alpha
    print("Krippendorff's alpha by category:")
    for cat in categories:
        alpha = alpha_by_cat.get(cat, float('nan'))
        print(cat, ': ', f'{alpha:.3f}')

if __name__ == '__main__':
    main()