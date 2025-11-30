#!/usr/bin/env python3
"""
Master script to extract all features (FMA + MERT + JMLA)
Orchestrates all three extraction scripts in sequence
"""

import time
import subprocess
import sys

def run_extraction(script_name, description):
    """Run extraction script and track time"""
    print(f"\n{'='*60}")
    print(f"Starting: {description}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        elapsed = time.time() - start_time
        
        print(f"\n✅ {description} completed in {elapsed/3600:.2f} hours")
        return True, elapsed
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description} failed: {e}")
        return False, 0

def main():
    """Extract all features sequentially"""
    print("="*60)
    print("Music_ReClass - Feature Extraction Pipeline")
    print("Processing 100k songs with 3 models")
    print("="*60)
    
    total_start = time.time()
    results = {}
    
    # Stage 1: FMA (CPU, fast)
    success, elapsed = run_extraction('extract_fma.py', 'FMA Features (518 dims)')
    results['FMA'] = {'success': success, 'time': elapsed}
    
    if not success:
        print("\n⚠️  FMA extraction failed. Continue anyway? (y/n)")
        if input().lower() != 'y':
            return
    
    # Stage 2: MERT (GPU, medium)
    success, elapsed = run_extraction('extract_mert.py', 'MERT Features (768 dims)')
    results['MERT'] = {'success': success, 'time': elapsed}
    
    if not success:
        print("\n⚠️  MERT extraction failed. Continue anyway? (y/n)")
        if input().lower() != 'y':
            return
    
    # Stage 3: JMLA (GPU, medium)
    success, elapsed = run_extraction('extract_jmla.py', 'JMLA Features (768 dims)')
    results['JMLA'] = {'success': success, 'time': elapsed}
    
    # Summary
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*60)
    print("EXTRACTION SUMMARY")
    print("="*60)
    
    for model, result in results.items():
        status = "✅" if result['success'] else "❌"
        hours = result['time'] / 3600
        print(f"{status} {model:6s}: {hours:6.2f} hours")
    
    print(f"\nTotal time: {total_elapsed/3600:.2f} hours ({total_elapsed/86400:.2f} days)")
    
    # Check completion
    all_success = all(r['success'] for r in results.values())
    if all_success:
        print("\n✅ All features extracted successfully!")
        print("Next step: Run train_extended.py to train the model")
    else:
        print("\n⚠️  Some extractions failed. Check logs above.")

if __name__ == '__main__':
    main()
