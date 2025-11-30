import time
import json
from pathlib import Path

results = {}

def run_xgboost():
    """Run XGBoost baseline"""
    print("\n" + "="*60)
    print("RUNNING XGBOOST BASELINE")
    print("="*60)
    
    start = time.time()
    import subprocess
    result = subprocess.run(['python3', 'train_xgboost_fma.py'], 
                          capture_output=True, text=True)
    elapsed = time.time() - start
    
    # Extract accuracy from output
    accuracy = None
    for line in result.stdout.split('\n'):
        if 'Test Accuracy:' in line:
            accuracy = float(line.split(':')[1].strip().replace('%', ''))
    
    results['xgboost'] = {
        'method': 'XGBoost + Pre-computed Features',
        'dataset': 'FMA Medium',
        'training_time': f"{elapsed/60:.1f} min",
        'accuracy': f"{accuracy:.2f}%" if accuracy else "N/A",
        'gpu_required': False,
        'model_size': 'Small (~MB)'
    }
    
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    return accuracy

def run_deep_learning():
    """Run OpenJMLA transfer learning"""
    print("\n" + "="*60)
    print("RUNNING DEEP LEARNING (OpenJMLA)")
    print("="*60)
    
    start = time.time()
    import subprocess
    result = subprocess.run(['python3', 'train_gtzan_openjmla.py'], 
                          capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - start
    
    # Extract final validation accuracy
    accuracy = None
    for line in result.stdout.split('\n'):
        if 'Val Acc:' in line:
            accuracy = float(line.split('Val Acc:')[1].strip().replace('%', ''))
    
    results['openjmla'] = {
        'method': 'OpenJMLA Transfer Learning',
        'dataset': 'GTZAN',
        'training_time': f"{elapsed/60:.1f} min",
        'accuracy': f"{accuracy:.2f}%" if accuracy else "N/A",
        'gpu_required': True,
        'model_size': 'Large (330MB + classifier)'
    }
    
    print(f"\nCompleted in {elapsed/60:.1f} minutes")
    return accuracy

def print_comparison():
    """Print comparison table"""
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Method':<35} {'Dataset':<12} {'Time':<10} {'Accuracy':<10} {'GPU'}")
    print("-"*70)
    
    for name, data in results.items():
        print(f"{data['method']:<35} {data['dataset']:<12} {data['training_time']:<10} "
              f"{data['accuracy']:<10} {'Yes' if data['gpu_required'] else 'No'}")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    print("• XGBoost: Fast baseline, no GPU needed, good for quick experiments")
    print("• OpenJMLA: Better accuracy potential, requires GPU, longer training")
    print("• For production: Start with XGBoost, upgrade to DL if needed")
    print("="*70)
    
    # Save results
    with open('model_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to model_comparison.json")

if __name__ == "__main__":
    print("Starting model comparison...")
    print("This will run both XGBoost and Deep Learning approaches")
    
    try:
        xgb_acc = run_xgboost()
        dl_acc = run_deep_learning()
        print_comparison()
    except KeyboardInterrupt:
        print("\n\nComparison interrupted by user")
        if results:
            print_comparison()
    except Exception as e:
        print(f"\nError during comparison: {e}")
        if results:
            print_comparison()
