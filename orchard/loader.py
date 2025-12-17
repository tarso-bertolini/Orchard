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
                # Parse key to find layer and type
                # e.g. model.layers.0.self_attn.q_proj.weight
                parts = key.split('.')
                
                if parts[0] == "model" and parts[1] == "layers":
                    layer_idx = int(parts[2])
                    layer = model.layers[layer_idx]
                    module = parts[3] # self_attn or mlp or input_layernorm
                    
                    tensor = f.get_tensor(key)
                    
                    # Quantize and Load
                    if module == "self_attn":
                        proj = parts[4] # q_proj, k_proj, v_proj, o_proj
                        if proj == "q_proj":
                            w, s = _quantize_and_upload(model.backend, tensor)
                            layer.wq_w = w; layer.wq_s = s
                        elif proj == "k_proj":
                            w, s = _quantize_and_upload(model.backend, tensor)
                            layer.wk_w = w; layer.wk_s = s
                        elif proj == "v_proj":
                            w, s = _quantize_and_upload(model.backend, tensor)
                            layer.wv_w = w; layer.wv_s = s
                        elif proj == "o_proj":
                            w, s = _quantize_and_upload(model.backend, tensor)
                            layer.wo_w = w; layer.wo_s = s
                            
                    elif module == "mlp":
                        proj = parts[4]
                        if proj == "gate_proj":
                            w, s = _quantize_and_upload(model.backend, tensor)
                            layer.w1_w = w; layer.w1_s = s
                        elif proj == "up_proj":
                            w, s = _quantize_and_upload(model.backend, tensor)
                            layer.w3_w = w; layer.w3_s = s
                        elif proj == "down_proj":
                            w, s = _quantize_and_upload(model.backend, tensor)
                            layer.w2_w = w; layer.w2_s = s
                            
                    elif module == "input_layernorm":
                        # FP32 upload
                        layer.attn_norm = _upload_fp32(model.backend, tensor)
                    elif module == "post_attention_layernorm":
                        layer.ffn_norm = _upload_fp32(model.backend, tensor)
                        
                elif parts[0] == "model" and parts[1] == "norm":
                    # Final norm
                    tensor = f.get_tensor(key)
                    model.norm = _upload_fp32(model.backend, tensor)
                    
                elif parts[0] == "lm_head":
                    # Output head
                    # Usually kept in FP16 or quantized? Let's quantize for memory
                    tensor = f.get_tensor(key)
                    w, s = _quantize_and_upload(model.backend, tensor)
                    model.output_w = w; model.output_s = s

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
