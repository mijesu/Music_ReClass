import torch
import numpy as np

class EarlyStopping:
    """Early stops training if validation loss/accuracy doesn't improve"""
    
    def __init__(self, patience=10, mode='max', delta=0, save_path='best_model.pth'):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            mode (str): 'min' for loss, 'max' for accuracy
            delta (float): Minimum change to qualify as improvement
            save_path (str): Path to save best model
        """
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.save_path = save_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, epoch, metric, model):
        """
        Call after each epoch
        
        Args:
            epoch (int): Current epoch number
            metric (float): Validation metric (loss or accuracy)
            model: PyTorch model to save
            
        Returns:
            bool: True if should stop training
        """
        score = metric if self.mode == 'max' else -metric
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(model)
            return False
        
        if score > self.best_score + self.delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self.save_checkpoint(model)
            print(f"  ✓ Best model saved (Epoch {epoch+1})")
            return False
        else:
            self.counter += 1
            print(f"  Patience: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"\n⚠ Early stopping triggered!")
                print(f"  Best epoch: {self.best_epoch + 1}")
                print(f"  Best {'accuracy' if self.mode == 'max' else 'loss'}: {self.best_score if self.mode == 'max' else -self.best_score:.4f}")
                return True
            
            return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        torch.save(model.state_dict(), self.save_path)
    
    def load_best_model(self, model):
        """Load best model"""
        model.load_state_dict(torch.load(self.save_path))
        return model
