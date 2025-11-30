import torch

model_path = "/media/mijesu_970/SSD_Data/AI_models/OpenJMLA/epoch_20.pth"
checkpoint = torch.load(model_path, map_location='cpu')

print(f"Model keys: {list(checkpoint.keys())}")
print(f"\nAuthor: {checkpoint.get('author', 'N/A')}")

if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    print(f"\nState dict has {len(state_dict)} parameters")
    print("\nFirst 10 parameter names:")
    for i, key in enumerate(list(state_dict.keys())[:10]):
        print(f"  {key}: {state_dict[key].shape}")
