import json
import os
import glob
import numpy as np
from safetensors import safe_open
from .quantizer import quantize_q4_0
import orchard_core

def load_weights(model, model_path):
    print(f"Loading model from {model_path}...")
    
    # 1. Load Config
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    
    model.config = config
    n_layers = config['num_hidden_layers']
    hidden_dim = config['hidden_size']
    
    # Initialize layers
    from .model import LlamaLayer
    model.layers = [LlamaLayer(model.backend, config, i) for i in range(n_layers)]
    
    # 2. Find Safetensors files
    files = glob.glob(os.path.join(model_path, "*.safetensors"))
    if not files:
        raise FileNotFoundError("No .safetensors files found")
    
    # 3. Load Weights
    # We iterate over files and load what we find
    for file in files:
        print(f"Processing {os.path.basename(file)}...")
        with safe_open(file, framework="numpy", device="cpu") as f:
            keys = f.keys()
            for key in keys:
                if key.endswith("_scales"):
                    continue # Handled by _packed
                
                # Parse key to find layer and type
                # e.g. model.layers.0.self_attn.q_proj.weight
                parts = key.split('.')
                is_packed = key.endswith("_packed")
                
                if parts[0] == "model" and parts[1] == "layers":
                    layer_idx = int(parts[2])
                    layer = model.layers[layer_idx]
                    module = parts[3] # self_attn or mlp or input_layernorm
                    
                    if module == "self_attn" or module == "mlp":
                        proj = parts[4]
                        target_w, target_s = None, None
                        
                        if module == "self_attn":
                            if proj == "q_proj": target_w, target_s = "wq_w", "wq_s"
                            elif proj == "k_proj": target_w, target_s = "wk_w", "wk_s"
                            elif proj == "v_proj": target_w, target_s = "wv_w", "wv_s"
                            elif proj == "o_proj": target_w, target_s = "wo_w", "wo_s"
                        elif module == "mlp":
                            if proj == "gate_proj": target_w, target_s = "w1_w", "w1_s"
                            elif proj == "up_proj": target_w, target_s = "w3_w", "w3_s"
                            elif proj == "down_proj": target_w, target_s = "w2_w", "w2_s"
                            
                        if target_w:
                            if is_packed:
                                packed = f.get_tensor(key)
                                scales = f.get_tensor(key.replace("_packed", "_scales"))
                                w, s = _upload_quantized(model.backend, packed, scales)
                            else:
                                tensor = f.get_tensor(key)
                                w, s = _quantize_and_upload(model.backend, tensor)
                            
                            setattr(layer, target_w, w)
                            setattr(layer, target_s, s)
                            
                    elif module == "input_layernorm":
                        # FP32 upload
                        layer.attn_norm = _upload_fp32(model.backend, f.get_tensor(key))
                    elif module == "post_attention_layernorm":
                        layer.ffn_norm = _upload_fp32(model.backend, f.get_tensor(key))
                        
                elif parts[0] == "model" and parts[1] == "norm":
                    # Final norm
                    tensor = f.get_tensor(key)
                    model.norm = _upload_fp32(model.backend, tensor)
                
                elif parts[0] == "model" and parts[1] == "embed_tokens":
                    # Embeddings
                    # Keep on CPU as numpy array for v0.1
                    tensor = f.get_tensor(key)
                    if tensor.dtype != np.float32:
                        tensor = tensor.astype(np.float32)
                    model.embed_tokens = tensor
                    
                elif parts[0] == "lm_head":
                    # Output head
                    if is_packed:
                        packed = f.get_tensor(key)
                        scales = f.get_tensor(key.replace("_packed", "_scales"))
                        w, s = _upload_quantized(model.backend, packed, scales)
                    else:
                        tensor = f.get_tensor(key)
                        w, s = _quantize_and_upload(model.backend, tensor)
                    model.output_w = w; model.output_s = s

def _upload_quantized(backend, packed, scales):
    rows, cols_packed = packed.shape
    t_w = orchard_core.Tensor(backend, [rows, cols_packed], orchard_core.DType.Int8)
    t_w.copy_from_host(packed)
    
    rows, cols_scales = scales.shape
    t_s = orchard_core.Tensor(backend, [rows, cols_scales], orchard_core.DType.Float16)
    t_s.copy_from_host(scales)
    
    return t_w, t_s

def _quantize_and_upload(backend, tensor_np):
    # tensor_np is (Out, In)
    # Quantize
    # Ensure float32 for quantization precision
    if tensor_np.dtype != np.float32:
        tensor_np = tensor_np.astype(np.float32)
        
    packed, scales = quantize_q4_0(tensor_np)
    
    # Upload
    # Packed: (N, K/2) uint8
    # Scales: (N, K/32) float16
    
    rows, cols_packed = packed.shape
    t_w = orchard_core.Tensor(backend, [rows, cols_packed], orchard_core.DType.Int8)
    t_w.copy_from_host(packed)
    
    rows, cols_scales = scales.shape
    t_s = orchard_core.Tensor(backend, [rows, cols_scales], orchard_core.DType.Float16)
    t_s.copy_from_host(scales)
    
    return t_w, t_s

def _upload_fp32(backend, tensor_np):
    # Convert to float32 if not already
    if tensor_np.dtype != np.float32:
        tensor_np = tensor_np.astype(np.float32)
        
    shape = tensor_np.shape
    t = orchard_core.Tensor(backend, list(shape), orchard_core.DType.Float32)
    t.copy_from_host(tensor_np)
    return t
