import os
import json
import glob
import numpy as np
from safetensors.numpy import save_file, load_file
from .quantizer import quantize_q4_0

def optimize_model(input_path: str, output_path: str):
    """
    Optimizes a model by pre-quantizing weights to Q4_0 and saving them.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    print(f"Optimizing model from {input_path} to {output_path}...")
    
    # 1. Copy Config & Tokenizer
    # We just copy non-safetensors files
    for filename in os.listdir(input_path):
        if not filename.endswith(".safetensors") and not os.path.isdir(os.path.join(input_path, filename)):
            # Simple copy
            with open(os.path.join(input_path, filename), 'rb') as src, open(os.path.join(output_path, filename), 'wb') as dst:
                dst.write(src.read())
                
    # 2. Process Safetensors
    files = glob.glob(os.path.join(input_path, "*.safetensors"))
    
    for file_path in files:
        filename = os.path.basename(file_path)
        print(f"Processing {filename}...")
        
        tensors = load_file(file_path)
        new_tensors = {}
        
        for key, tensor in tensors.items():
            # Check if this is a weight we want to quantize
            # We quantize Linear layers in the transformer blocks and the output head
            # Keys usually look like: model.layers.0.self_attn.q_proj.weight
            
            should_quantize = False
            if "model.layers" in key and "weight" in key and "norm" not in key:
                # Attention and MLP weights
                should_quantize = True
            elif "lm_head.weight" in key:
                should_quantize = True
                
            if should_quantize:
                print(f"  Quantizing {key}...")
                # Ensure float32
                if tensor.dtype != np.float32:
                    tensor = tensor.astype(np.float32)
                    
                packed, scales = quantize_q4_0(tensor)
                
                new_tensors[key.replace(".weight", ".weight_packed")] = packed
                new_tensors[key.replace(".weight", ".weight_scales")] = scales
            else:
                # Keep as is (e.g. norms, embeddings)
                print(f"  Copying {key}...")
                new_tensors[key] = tensor
                
        # Save new file
        save_path = os.path.join(output_path, filename)
        save_file(new_tensors, save_path)
        
    print("Optimization complete.")
