import torch
import subprocess

def get_gpu_memory():
    """Get GPU memory usage in MB"""
    if not torch.cuda.is_available():
        return None
    
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'],
            encoding='utf-8'
        )
        used, total = map(int, result.strip().split(','))
        return {'used': used, 'total': total, 'free': total - used, 'percent': (used/total)*100}
    except:
        # Fallback to torch
        used = torch.cuda.memory_allocated() / 1024**2
        total = torch.cuda.get_device_properties(0).total_memory / 1024**2
        return {'used': used, 'total': total, 'free': total - used, 'percent': (used/total)*100}

def suggest_batch_size(free_memory_mb, sample_size_mb=100):
    """Suggest batch size based on available GPU memory"""
    safe_memory = free_memory_mb * 0.7  # Use 70% of free memory
    return max(1, int(safe_memory / sample_size_mb))

def print_gpu_status():
    """Print current GPU memory status"""
    mem = get_gpu_memory()
    if mem:
        print(f"GPU Memory: {mem['used']:.0f}MB / {mem['total']:.0f}MB ({mem['percent']:.1f}%)")
        print(f"Free: {mem['free']:.0f}MB")
        print(f"Suggested batch size: {suggest_batch_size(mem['free'])}")
    else:
        print("No GPU available")

if __name__ == "__main__":
    print_gpu_status()
