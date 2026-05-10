"""
Script to execute all notebooks sequentially and generate the final report.
"""

import subprocess
import sys
import os

os.chdir('c:\\Users\\ASUS\\Documents\\Evan\\Coding\\College\\ML\\tubes')

notebooks = [
    'notebooks/01_preprocessing.ipynb',
    'notebooks/02_pca_kmeans.ipynb',
    'notebooks/03_supervised.ipynb',
    'notebooks/04_evaluation.ipynb'
]

print("="*80)
print("RUNNING ML PIPELINE NOTEBOOKS")
print("="*80)

for i, notebook in enumerate(notebooks, 1):
    print(f"\n[{i}/{len(notebooks)}] Running {notebook}...")
    print("-"*80)
    
    cmd = [
        sys.executable, 
        '-m', 
        'nbconvert', 
        '--to', 
        'notebook', 
        '--execute',
        '--inplace',
        notebook
    ]
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"⚠ Warning: Notebook {notebook} had exit code {result.returncode}")
    else:
        print(f"✓ Completed {notebook}")

print("\n" + "="*80)
print("ALL NOTEBOOKS EXECUTED!")
print("="*80)
