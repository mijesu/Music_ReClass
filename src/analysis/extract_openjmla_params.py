import torch
import json

MODEL_PATH = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"

def extract_openjmla_parameters():
    """Extract OpenJMLA model parameters and architecture"""
    
    print("🔍 OpenJMLA Model Parameter Extraction\n")
    print("=" * 60)
    
    # Load model
    print(f"Loading: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location='cpu')
    
    print(f"✅ Model loaded\n")
    
    # Basic info
    print("📋 Model Information:")
    print(f"   Keys: {list(checkpoint.keys())}")
    print(f"   Author: {checkpoint.get('author', 'N/A')}\n")
    
    # Get state dict
    state_dict = checkpoint.get('state_dict', checkpoint)
    
    # Count parameters
    total_params = 0
    trainable_params = 0
    
    print("📊 Model Architecture:\n")
    print(f"   Total layers: {len(state_dict)}")
    
    # Analyze structure
    layer_info = {}
    for name, param in state_dict.items():
        total_params += param.numel()
        
        # Group by layer type
        layer_type = name.split('.')[0]
        if layer_type not in layer_info:
            layer_info[layer_type] = {
                'count': 0,
                'params': 0,
                'shapes': []
            }
        layer_info[layer_type]['count'] += 1
        layer_info[layer_type]['params'] += param.numel()
        layer_info[layer_type]['shapes'].append(list(param.shape))
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: {total_params * 4 / 1024 / 1024:.2f} MB (float32)\n")
    
    # Layer breakdown
    print("🏗️  Layer Breakdown:")
    for layer_type, info in sorted(layer_info.items()):
        print(f"\n   {layer_type}:")
        print(f"      Count: {info['count']}")
        print(f"      Parameters: {info['params']:,}")
        print(f"      Example shapes: {info['shapes'][:3]}")
    
    # Key architecture details
    print("\n\n🎯 Key Architecture Details:")
    
    # Check for Vision Transformer components
    if 'pos_embed' in state_dict:
        pos_embed_shape = state_dict['pos_embed'].shape
        print(f"   Position Embedding: {list(pos_embed_shape)}")
        print(f"      Sequence length: {pos_embed_shape[1]}")
        print(f"      Embedding dim: {pos_embed_shape[2]}")
    
    if 'cls_token' in state_dict:
        cls_shape = state_dict['cls_token'].shape
        print(f"   CLS Token: {list(cls_shape)}")
    
    if 'patch_embed.projection.weight' in state_dict:
        patch_shape = state_dict['patch_embed.projection.weight'].shape
        print(f"   Patch Embedding: {list(patch_shape)}")
        print(f"      Patch size: {patch_shape[2]}x{patch_shape[3]}")
        print(f"      Embedding dim: {patch_shape[0]}")
    
    # Count attention layers
    attn_layers = [k for k in state_dict.keys() if 'attn' in k]
    num_attn_layers = len(set([k.split('.')[1] for k in attn_layers if 'layers' in k]))
    if num_attn_layers > 0:
        print(f"   Transformer layers: {num_attn_layers}")
    
    # Export parameters to JSON
    export_data = {
        'model_name': 'OpenJMLA',
        'author': checkpoint.get('author', 'MMSelfSup'),
        'total_parameters': total_params,
        'total_layers': len(state_dict),
        'model_size_mb': round(total_params * 4 / 1024 / 1024, 2),
        'architecture': {
            'type': 'Vision Transformer (ViT)',
            'embedding_dim': state_dict['pos_embed'].shape[2] if 'pos_embed' in state_dict else None,
            'sequence_length': state_dict['pos_embed'].shape[1] if 'pos_embed' in state_dict else None,
            'num_layers': num_attn_layers if num_attn_layers > 0 else None
        },
        'layer_breakdown': {k: {'count': v['count'], 'params': v['params']} for k, v in layer_info.items()}
    }
    
    output_file = 'openjmla_parameters.json'
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"\n\n💾 Parameters exported to: {output_file}")
    print("=" * 60)
    
    return export_data

if __name__ == '__main__':
    params = extract_openjmla_parameters()
