# correct.py
import os
from rapidfuzz import process, fuzz

# Ensure medicines.txt is loaded relative to this script
base_dir = os.path.dirname(__file__)
medicines_path = os.path.join(base_dir, 'medicines.txt')

with open(medicines_path, 'r') as f:
    DICTIONARY = [line.strip() for line in f if line.strip()]

def correct(raw_pred, threshold_accept=75, threshold_uncertain=50):
    if not raw_pred or raw_pred.strip() == '':
        return raw_pred, 0, 'unrecognized'
   
    result = process.extractOne(
        raw_pred.lower(),
        DICTIONARY,
        scorer=fuzz.ratio
    )
   
    if result is None:
        return raw_pred, 0, 'unrecognized'
   
    match, score, _ = result
    if score >= threshold_accept:
        return match, score, 'corrected'
    elif score >= threshold_uncertain:
        return match, score, 'uncertain'
    else:
        return raw_pred, score, 'unrecognized'


if __name__ == '__main__':
    # Test with your actual eval results
    test_cases = [
        ('amphogep',     'amphogel'),
        ('corestin',     'corestin'),
        ('amldipine',    'amlodipine'),
        ('axcer',        'axcer'),
        ('alpzole',      'omeprazole'),
        ('brnzox',       'brinzox'),
        ('sartl',        'sartel'),
        ('oligel',       'oligel'),
        ('montt-gtn',    'monit-gtn'),
        ('pacimel-lmf',  'pacimol-mf'),
        ('golvin-seld',  'solvin-cold'),
        ('ketarol',      'ketorol'),
        ('aligel',       'oligel'),
        ('tibonor',      'tibonor'),
        ('stamlo',       'stamlo'),
    ]

    print(f"{'RAW PRED':<20} {'CORRECTED':<20} {'GT':<20} {'SCORE':<8} STATUS")
    print('-' * 80)
   
    raw_correct = 0
    after_correct = 0
   
    for raw, gt in test_cases:
        corrected, score, status = correct(raw)
        raw_ok  = '✓' if raw == gt else '✗'
        post_ok = '✓' if corrected == gt else '✗'
        if raw == gt: raw_correct += 1
        if corrected == gt: after_correct += 1
        print(f"{raw_ok} {raw:<19} {post_ok} {corrected:<19} {gt:<20} {score:<8.1f} {status}")
   
    print(f"\nBefore correction: {raw_correct}/{len(test_cases)} correct")
    print(f"After  correction: {after_correct}/{len(test_cases)} correct")
