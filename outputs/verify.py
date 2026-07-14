#!/usr/bin/env python3
"""Verification script: recomputes ARI from saved MCFST predictions."""
import os, glob, json, numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_PATH = os.path.join(ROOT, 'preprocessed_datav1', 'input', 'label.npy')
RESULT_DIR = os.path.join(ROOT, 'outputs', 'result')

def main():
    label = np.load(LABEL_PATH)
    print(f"Ground truth: {len(label)} spots, {len(np.unique(label))} clusters")
    
    files = sorted(glob.glob(os.path.join(RESULT_DIR, 'ari_*.npy')))
    if not files:
        print("ERROR: No prediction files found!")
        return
    
    results = []
    for f in files:
        pred = np.load(f)
        ari = adjusted_rand_score(label, pred)
        expected_ari = float(os.path.basename(f).replace('ari_', '').replace('.npy', ''))
        match = "✓" if abs(ari - expected_ari) < 1e-6 else "✗ MISMATCH"
        results.append({
            'file': os.path.basename(f),
            'expected_ari': expected_ari,
            'recomputed_ari': ari,
            'match': match == '✓',
        })
        print(f"  {os.path.basename(f)}: expected={expected_ari:.4f}, recomputed={ari:.4f} {match}")
    
    aris = [r['recomputed_ari'] for r in results]
    print(f"\n--- Summary ---")
    print(f"n_runs: {len(aris)}")
    print(f"Mean ± Std: {np.mean(aris):.4f} ± {np.std(aris, ddof=1):.4f}")
    print(f"Median: {np.median(aris):.4f}")
    print(f"Min: {np.min(aris):.4f}, Max: {np.max(aris):.4f}")
    print(f"Paper-reported ARI: 0.693")
    print(f"Best reproduced ARI: {np.max(aris):.4f}")
    
    # Verify all match
    all_match = all(r['match'] for r in results)
    print(f"\nAll predictions match expected ARI: {all_match}")
    
    with open(os.path.join(ROOT, 'outputs', 'verified_metrics.json'), 'w') as f:
        json.dump({
            'n_runs': len(aris),
            'aris': [float(x) for x in aris],
            'mean': float(np.mean(aris)),
            'std': float(np.std(aris, ddof=1)),
            'median': float(np.median(aris)),
            'min': float(np.min(aris)),
            'max': float(np.max(aris)),
            'paper_reported': 0.693,
            'best_reproduced': float(np.max(aris)),
            'all_verified': all_match,
        }, f, indent=2)
    print(f"Verified metrics saved to outputs/verified_metrics.json")

if __name__ == '__main__':
    main()
