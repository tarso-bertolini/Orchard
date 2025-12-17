import orchard_core
import numpy as np
import os
from typing import List, Optional, Tuple

class LlamaLayer:
    def __init__(self, backend, config, layer_idx):
        self.backend = backend
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_dim = config['hidden_size']
        self.heads = config['num_attention_heads']
        self.head_dim = self.hidden_dim // self.heads
        self.intermediate_dim = config['intermediate_size']
        
        # Weights (Quantized)
        # We hold them as None initially, loaded later
        self.wq_w = None; self.wq_s = None
        self.wk_w = None; self.wk_s = None
        self.wv_w = None; self.wv_s = None
        self.wo_w = None; self.wo_s = None
        
        self.w1_w = None; self.w1_s = None # Gate
        self.w2_w = None; self.w2_s = None # Down
        self.w3_w = None; self.w3_s = None # Up
        
        self.attn_norm = None
        self.ffn_norm = None

    def forward(self, x, freqs_cos, freqs_sin):
        # x: [Hidden] (Tensor)
        # This is a simplified forward for Batch=1
        
        # 1. RMSNorm
        # We need a temp tensor for output of norm
        # For now, let's assume x is on GPU.
        
        # TODO: Full implementation requires managing temp buffers
        # and the missing element-wise kernels (Add, Silu, Softmax).
        # For this v0.1, we will implement the structure but 
        # the actual run loop will be in the main generation function
        # to allow for easier hacking of the CPU/GPU boundary.
        pass

class Llama:
    def __init__(self, model_path: str):
        self.backend = orchard_core.MetalBackend()
        self.backend.initialize()
        if not self.backend.is_available():
            raise RuntimeError("Metal backend not available")
            
        print(f"Orchard initialized on {self.backend.get_device_name()}")
        
        self.layers: List[LlamaLayer] = []
        self.load_model(model_path)
        
    def load_model(self, model_path: str):
        from .loader import load_weights
        # This will populate self.layers and other params
        load_weights(self, model_path)
        
    def generate(self, prompt: str, max_tokens: int = 50):
        # Tokenizer logic here
        pass
