import torch
import sys
from pathlib import Path

def check_model_compatibility(model_path):
    """Check if a trained model is compatible with current system"""
    
    print("🔍 Model Compatibility Check\n")
    print("=" * 50)
    
    # Check if file exists
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    print(f"✅ Model file found: {model_path}")
    print(f"   Size: {Path(model_path).stat().st_size / 1024:.2f} KB\n")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Try to load model
    try:
        print("\n📂 Loading model...")
        checkpoint = torch.load(model_path, map_location='cpu')
        print("✅ Model loaded successfully")
        
        # Check if it's a state dict or full checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                print("   Type: Training checkpoint")
                state_dict = checkpoint['model_state_dict']
                print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
                print(f"   Loss: {checkpoint.get('loss', 'N/A')}")
                print(f"   Accuracy: {checkpoint.get('accuracy', 'N/A')}")
            else:
                print("   Type: Model state dict")
                state_dict = checkpoint
        else:
            print("   Type: Full model")
            state_dict = checkpoint.state_dict()
        
        # Check model structure
        print(f"\n📊 Model Structure:")
        print(f"   Total parameters: {len(state_dict)}")
        print(f"   First 5 layers:")
        for i, (name, param) in enumerate(list(state_dict.items())[:5]):
            print(f"      {name}: {list(param.shape)}")
        
        # Check CUDA compatibility
        print(f"\n🖥️  System Compatibility:")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # Try to instantiate model
        print(f"\n🧪 Testing model instantiation...")
        from Classifed_JMLA_GTZAN import GenreClassifier
        
        model = GenreClassifier(num_classes=10)
        model.load_state_dict(state_dict if not isinstance(checkpoint, dict) or 'model_state_dict' not in checkpoint else checkpoint['model_state_dict'])
        model.eval()
        print("✅ Model instantiated successfully")
        
        # Test inference
        print(f"\n🎯 Testing inference...")
        dummy_input = torch.randn(1, 1, 128, 1292)  # Batch, Channel, Mel-bands, Time
        with torch.no_grad():
            output = model(dummy_input)
        print(f"✅ Inference test passed")
        print(f"   Input shape: {list(dummy_input.shape)}")
        print(f"   Output shape: {list(output.shape)}")
        print(f"   Output classes: {output.shape[1]}")
        
        print("\n" + "=" * 50)
        print("✅ MODEL IS COMPATIBLE!")
        print("=" * 50)
        return True
        
    except Exception as e:
        print(f"\n❌ Compatibility check failed!")
        print(f"   Error: {str(e)}")
        print("\n" + "=" * 50)
        print("❌ MODEL IS NOT COMPATIBLE")
        print("=" * 50)
        return False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = 'genre_classifier.pth'
    
    check_model_compatibility(model_path)
