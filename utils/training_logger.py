import json
import csv
from pathlib import Path
from datetime import datetime

class TrainingLogger:
    """Log training parameters and metrics to file"""
    
    def __init__(self, log_dir='./logs', experiment_name=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = experiment_name or f'train_{timestamp}'
        
        self.config_file = self.log_dir / f'{self.experiment_name}_config.json'
        self.metrics_file = self.log_dir / f'{self.experiment_name}_metrics.csv'
        self.log_file = self.log_dir / f'{self.experiment_name}.log'
        
        self.config = {}
        self.metrics = []
        
        # Initialize CSV
        with open(self.metrics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr', 'time'])
    
    def log_config(self, **kwargs):
        """Log training configuration"""
        self.config.update(kwargs)
        self.config['timestamp'] = datetime.now().isoformat()
        
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        self._write_log(f"\n{'='*70}")
        self._write_log(f"TRAINING CONFIGURATION - {self.experiment_name}")
        self._write_log(f"{'='*70}")
        for key, value in self.config.items():
            self._write_log(f"{key}: {value}")
        self._write_log(f"{'='*70}\n")
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        """Log metrics for one epoch"""
        metrics = {
            'epoch': epoch + 1,
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.2f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}',
            'lr': f'{lr:.6f}',
            'time': f'{epoch_time:.1f}'
        }
        
        self.metrics.append(metrics)
        
        # Append to CSV
        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(metrics.values())
        
        # Write to log
        log_msg = (f"Epoch {metrics['epoch']} ({metrics['time']}s) - "
                   f"Train: {metrics['train_loss']}/{metrics['train_acc']}% | "
                   f"Val: {metrics['val_loss']}/{metrics['val_acc']}% | "
                   f"LR: {metrics['lr']}")
        self._write_log(log_msg)
    
    def log_message(self, message):
        """Log custom message"""
        self._write_log(message)
    
    def log_final_results(self, best_acc, total_time, classification_report=None):
        """Log final training results"""
        self._write_log(f"\n{'='*70}")
        self._write_log("TRAINING COMPLETE")
        self._write_log(f"{'='*70}")
        self._write_log(f"Best Validation Accuracy: {best_acc:.2f}%")
        self._write_log(f"Total Training Time: {total_time/3600:.2f} hours")
        
        if classification_report:
            self._write_log("\nClassification Report:")
            self._write_log(classification_report)
        
        self._write_log(f"{'='*70}\n")
        
        # Save summary
        summary = {
            'best_accuracy': best_acc,
            'total_time_hours': total_time / 3600,
            'total_epochs': len(self.metrics),
            'final_timestamp': datetime.now().isoformat()
        }
        
        summary_file = self.log_dir / f'{self.experiment_name}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _write_log(self, message):
        """Write message to log file and print"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line + '\n')
        
        print(message)
    
    def get_best_epoch(self):
        """Get epoch with best validation accuracy"""
        if not self.metrics:
            return None
        
        best_idx = max(range(len(self.metrics)), 
                      key=lambda i: float(self.metrics[i]['val_acc']))
        return self.metrics[best_idx]
