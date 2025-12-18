import orchard_core
import numpy as np
import os
import time
from typing import List, Optional, Tuple

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (np.arange(0, dim, 2)[: (dim // 2)].astype(np.float32) / dim))
    t = np.arange(end).astype(np.float32)
    freqs = np.outer(t, freqs)
    freqs_cos = np.cos(freqs)
    freqs_sin = np.sin(freqs)
    return freqs_cos, freqs_sin

class LoraAdapter:
    def __init__(self, backend, r, alpha, scaling):
        self.backend = backend
        self.r = r
        self.alpha = alpha
        self.scaling = scaling
        # Stored in format suitable for matmul:
        # A: [in_dim, r]
        # B: [r, out_dim]
        self.lora_A = None 
        self.lora_B = None

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
        self.wq_w = None; self.wq_s = None
        self.wk_w = None; self.wk_s = None
        self.wv_w = None; self.wv_s = None
        self.wo_w = None; self.wo_s = None
        
        self.w1_w = None; self.w1_s = None # Gate
        self.w2_w = None; self.w2_s = None # Down
        self.w3_w = None; self.w3_s = None # Up
        
        self.attn_norm = None
        self.ffn_norm = None
        
        # LoRA Adapters (key: module name, value: LoraAdapter)
        # Modules: wq, wk, wv, wo, w1, w2, w3
        self.lora_adapters = {}
        
        # KV Cache (CPU numpy arrays, uploaded on demand)
        # List of (Heads, HeadDim, SeqLen) per sequence
        # We assume a max batch size of 32 for now
        self.k_cache_np = [np.zeros((self.heads, self.head_dim, 0), dtype=np.float32) for _ in range(32)]
        self.v_cache_np = [np.zeros((self.heads, 0, self.head_dim), dtype=np.float32) for _ in range(32)]

    def _apply_lora(self, x, output, module_name, batch_size, in_dim, out_dim):
        if module_name in self.lora_adapters:
            adapter = self.lora_adapters[module_name]
            # x: [Batch, In]
            # A: [In, r]
            # B: [r, Out]
            
            # Temp = x @ A
            temp = orchard_core.Tensor(self.backend, [batch_size * adapter.r], orchard_core.DType.Float32)
            self.backend.run_matmul(x, adapter.lora_A, temp, batch_size, adapter.r, in_dim)
            
            # Result = Temp @ B
            lora_out = orchard_core.Tensor(self.backend, [batch_size * out_dim], orchard_core.DType.Float32)
            self.backend.run_matmul(temp, adapter.lora_B, lora_out, batch_size, out_dim, adapter.r)
            
            # Add to output (B is pre-scaled)
            self.backend.run_add(output, lora_out, output, batch_size * out_dim)

    def forward(self, x, freqs_cos, freqs_sin, batch_indices):
        # x: [Batch, Hidden] (Tensor)
        # freqs_cos/sin: [Batch, HeadDim/2] (Tensor) - RoPE for current positions
        # batch_indices: List[int] - Indices into the KV cache for each item in the batch
        
        batch_size = len(batch_indices)
        
        # 1. RMSNorm
        x_norm = orchard_core.Tensor(self.backend, [batch_size * self.hidden_dim], orchard_core.DType.Float32)
        self.backend.run_rmsnorm(x, self.attn_norm, x_norm, self.config['rms_norm_eps'], self.hidden_dim, batch_size)
        
        # 2. Q, K, V Projections
        q = orchard_core.Tensor(self.backend, [batch_size * self.hidden_dim], orchard_core.DType.Float32)
        k = orchard_core.Tensor(self.backend, [batch_size * self.hidden_dim], orchard_core.DType.Float32)
        v = orchard_core.Tensor(self.backend, [batch_size * self.hidden_dim], orchard_core.DType.Float32)
        
        if batch_size == 1:
            self.backend.run_gemv_q4_0(self.wq_w, self.wq_s, x_norm, q, self.hidden_dim, self.hidden_dim)
            self.backend.run_gemv_q4_0(self.wk_w, self.wk_s, x_norm, k, self.hidden_dim, self.hidden_dim)
            self.backend.run_gemv_q4_0(self.wv_w, self.wv_s, x_norm, v, self.hidden_dim, self.hidden_dim)
        else:
            self.backend.run_gemm_q4_0(self.wq_w, self.wq_s, x_norm, q, self.hidden_dim, self.hidden_dim, batch_size)
            self.backend.run_gemm_q4_0(self.wk_w, self.wk_s, x_norm, k, self.hidden_dim, self.hidden_dim, batch_size)
            self.backend.run_gemm_q4_0(self.wv_w, self.wv_s, x_norm, v, self.hidden_dim, self.hidden_dim, batch_size)
            
        # Apply LoRA
        self._apply_lora(x_norm, q, "wq", batch_size, self.hidden_dim, self.hidden_dim)
        self._apply_lora(x_norm, k, "wk", batch_size, self.hidden_dim, self.hidden_dim)
        self._apply_lora(x_norm, v, "wv", batch_size, self.hidden_dim, self.hidden_dim)
        
        # 3. RoPE
        self.backend.run_rope(q, freqs_cos, freqs_sin, q, self.head_dim, self.heads, batch_size)
        self.backend.run_rope(k, freqs_cos, freqs_sin, k, self.head_dim, self.heads, batch_size)
        
        # 4. Update KV Cache & Attention
        q_cpu = np.array(q.copy_to_host(), copy=False).reshape(batch_size, self.heads, self.head_dim)
        k_cpu = np.array(k.copy_to_host(), copy=False).reshape(batch_size, self.heads, self.head_dim)
        v_cpu = np.array(v.copy_to_host(), copy=False).reshape(batch_size, self.heads, self.head_dim)
        
        output_cpu = np.zeros((batch_size, self.heads, self.head_dim), dtype=np.float32)
        scale = 1.0 / np.sqrt(self.head_dim)
        
        for i, b_idx in enumerate(batch_indices):
            # Update Cache for this sequence (b_idx)
            # k_cpu[i]: (Heads, HeadDim)
            self.k_cache_np[b_idx] = np.dstack([self.k_cache_np[b_idx], k_cpu[i][:, :, None]])
            self.v_cache_np[b_idx] = np.hstack([self.v_cache_np[b_idx], v_cpu[i][:, None, :]])
            
            # Attention (Per Head)
            for h in range(self.heads):
                qs = q_cpu[i, h] # (HeadDim)
                ks = self.k_cache_np[b_idx][h] # (HeadDim, Seq)
                vs = self.v_cache_np[b_idx][h] # (Seq, HeadDim)
                
                score = np.matmul(qs, ks) * scale
                score_max = np.max(score)
                exp_score = np.exp(score - score_max)
                softmax = exp_score / np.sum(exp_score)
                
                output_cpu[i, h] = np.matmul(softmax, vs)
            
        # Flatten output
        output_flat = output_cpu.flatten()
        
        # Upload back to GPU
        attn_output = orchard_core.Tensor(self.backend, [batch_size * self.hidden_dim], orchard_core.DType.Float32)
        attn_output.copy_from_host(output_flat)
        
        # 5. Output Projection
        attn_proj = orchard_core.Tensor(self.backend, [batch_size * self.hidden_dim], orchard_core.DType.Float32)
        if batch_size == 1:
            self.backend.run_gemv_q4_0(self.wo_w, self.wo_s, attn_output, attn_proj, self.hidden_dim, self.hidden_dim)
        else:
            self.backend.run_gemm_q4_0(self.wo_w, self.wo_s, attn_output, attn_proj, self.hidden_dim, self.hidden_dim, batch_size)
            
        # Apply LoRA
        self._apply_lora(attn_output, attn_proj, "wo", batch_size, self.hidden_dim, self.hidden_dim)
        
        # 6. Residual Add
        self.backend.run_add(x, attn_proj, x, batch_size * self.hidden_dim)
        
        # 7. FFN
        ffn_input = orchard_core.Tensor(self.backend, [batch_size * self.hidden_dim], orchard_core.DType.Float32)
        self.backend.run_rmsnorm(x, self.ffn_norm, ffn_input, self.config['rms_norm_eps'], self.hidden_dim, batch_size)
        
        # Gate & Up
        gate = orchard_core.Tensor(self.backend, [batch_size * self.intermediate_dim], orchard_core.DType.Float32)
        up = orchard_core.Tensor(self.backend, [batch_size * self.intermediate_dim], orchard_core.DType.Float32)
        
        if batch_size == 1:
            self.backend.run_gemv_q4_0(self.w1_w, self.w1_s, ffn_input, gate, self.hidden_dim, self.intermediate_dim)
            self.backend.run_gemv_q4_0(self.w3_w, self.w3_s, ffn_input, up, self.hidden_dim, self.intermediate_dim)
        else:
            self.backend.run_gemm_q4_0(self.w1_w, self.w1_s, ffn_input, gate, self.hidden_dim, self.intermediate_dim, batch_size)
            self.backend.run_gemm_q4_0(self.w3_w, self.w3_s, ffn_input, up, self.hidden_dim, self.intermediate_dim, batch_size)
            
        # Apply LoRA
        self._apply_lora(ffn_input, gate, "w1", batch_size, self.hidden_dim, self.intermediate_dim)
        self._apply_lora(ffn_input, up, "w3", batch_size, self.hidden_dim, self.intermediate_dim)
        
        # Silu & Mul
        self.backend.run_silu(gate, gate, batch_size * self.intermediate_dim)
        self.backend.run_mul(gate, up, gate, batch_size * self.intermediate_dim)
        
        # Down
        ffn_output = orchard_core.Tensor(self.backend, [batch_size * self.hidden_dim], orchard_core.DType.Float32)
        if batch_size == 1:
            self.backend.run_gemv_q4_0(self.w2_w, self.w2_s, gate, ffn_output, self.intermediate_dim, self.hidden_dim)
        else:
            self.backend.run_gemm_q4_0(self.w2_w, self.w2_s, gate, ffn_output, self.intermediate_dim, self.hidden_dim, batch_size)
            
        # Apply LoRA
        self._apply_lora(gate, ffn_output, "w2", batch_size, self.intermediate_dim, self.hidden_dim)
        
        # Residual Add
        self.backend.run_add(x, ffn_output, x, batch_size * self.hidden_dim)
        
        return x

class Llama:
    def __init__(self, model_path: str):
        self.backend = orchard_core.MetalBackend()
        
        # Find kernels path
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kernels_path = os.path.join(current_dir, "kernels")
        
        self.backend.initialize(kernels_path)
        if not self.backend.is_available():
            raise RuntimeError("Metal backend not available")
            
        print(f"Orchard initialized on {self.backend.get_device_name()}")
        
        self.layers: List[LlamaLayer] = []
        self.config = {}
        self.load_model(model_path)
        
        # Precompute RoPE
        self.freqs_cos, self.freqs_sin = precompute_freqs_cis(
            self.config['hidden_size'] // self.config['num_attention_heads'],
            self.config.get('max_position_embeddings', 2048)
        )
        
        # Load Tokenizer
        try:
            from tokenizers import Tokenizer
            tokenizer_path = os.path.join(model_path, "tokenizer.json")
            if os.path.exists(tokenizer_path):
                self.tokenizer = Tokenizer.from_file(tokenizer_path)
            else:
                # Fallback or error
                print("Warning: tokenizer.json not found. Generation will fail if not provided manually.")
                self.tokenizer = None
        except ImportError:
            print("Warning: 'tokenizers' library not found. Install it with `pip install tokenizers`.")
            self.tokenizer = None
        
    def load_model(self, model_path: str):
        from .loader import load_weights
        load_weights(self, model_path)
        
    def load_lora(self, adapter_path: str):
        from .loader import load_lora_adapter
        load_lora_adapter(self, adapter_path)
        
    def _sample(self, logits, temperature=0.7):
        if temperature == 0:
            return np.argmax(logits)
        else:
            # Softmax
            # Shift logits for stability
            logits = logits - np.max(logits)
            probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
            return np.random.choice(len(probs), p=probs)

    def generate_batch(self, prompts: List[str], max_tokens: int = 50, temperature: float = 0.7):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
            
        batch_size = len(prompts)
        
        # Tokenize
        batch_ids = [self.tokenizer.encode(p).ids for p in prompts]
        
        # Current position for each sequence
        # We need to track this because sequences might have different lengths
        # But for v0.1 we assume we process them in lockstep or just track pos.
        
        # Prefill
        # We process each prompt token by token to fill the cache.
        # Ideally we'd do this in parallel too, but for now let's just fill it up.
        
        print(f"Prefilling {batch_size} prompts...")
        current_tokens = []
        
        # We need to reset cache for these slots?
        # Assuming slots 0..batch_size-1 are free or we overwrite them.
        # The cache is just a list of numpy arrays, so we can just reset them.
        for b in range(batch_size):
            self.layers[0].k_cache_np[b] = np.zeros((self.layers[0].heads, self.layers[0].head_dim, 0), dtype=np.float32)
            self.layers[0].v_cache_np[b] = np.zeros((self.layers[0].heads, 0, self.layers[0].head_dim), dtype=np.float32)
            # We need to do this for all layers!
            for layer in self.layers:
                layer.k_cache_np[b] = np.zeros((layer.heads, layer.head_dim, 0), dtype=np.float32)
                layer.v_cache_np[b] = np.zeros((layer.heads, 0, layer.head_dim), dtype=np.float32)

        for b in range(batch_size):
            ids = batch_ids[b]
            for i, token_id in enumerate(ids):
                # Prepare input
                x_emb = self.embed_tokens[token_id]
                x = orchard_core.Tensor(self.backend, [self.config['hidden_size']], orchard_core.DType.Float32)
                x.copy_from_host(x_emb)
                
                # RoPE
                freqs_cos = self.freqs_cos[i]
                freqs_sin = self.freqs_sin[i]
                
                f_cos = orchard_core.Tensor(self.backend, [self.config['hidden_size'] // self.config['num_attention_heads'] // 2], orchard_core.DType.Float32)
                f_sin = orchard_core.Tensor(self.backend, [self.config['hidden_size'] // self.config['num_attention_heads'] // 2], orchard_core.DType.Float32)
                f_cos.copy_from_host(freqs_cos)
                f_sin.copy_from_host(freqs_sin)
                
                for layer in self.layers:
                    layer.forward(x, f_cos, f_sin, [b])
                    
            current_tokens.append(ids[-1])
            
        # Generation Loop
        print(f"Generating...")
        
        generated_sequences = [[] for _ in range(batch_size)]
        active_indices = list(range(batch_size))
        
        for step in range(max_tokens):
            if not active_indices:
                break
                
            # Prepare batch input
            curr_batch_size = len(active_indices)
            x_batch_host = np.zeros((curr_batch_size, self.config['hidden_size']), dtype=np.float32)
            
            head_dim = self.config['hidden_size'] // self.config['num_attention_heads']
            f_cos_host = np.zeros((curr_batch_size, head_dim // 2), dtype=np.float32)
            f_sin_host = np.zeros((curr_batch_size, head_dim // 2), dtype=np.float32)
            
            for i, b_idx in enumerate(active_indices):
                token_id = current_tokens[b_idx]
                x_batch_host[i] = self.embed_tokens[token_id]
                
                pos = len(batch_ids[b_idx]) + len(generated_sequences[b_idx])
                f_cos_host[i] = self.freqs_cos[pos]
                f_sin_host[i] = self.freqs_sin[pos]
                
            # Upload to GPU
            x_tensor = orchard_core.Tensor(self.backend, [curr_batch_size * self.config['hidden_size']], orchard_core.DType.Float32)
            x_tensor.copy_from_host(x_batch_host.flatten())
            
            f_cos_tensor = orchard_core.Tensor(self.backend, [curr_batch_size * (head_dim // 2)], orchard_core.DType.Float32)
            f_cos_tensor.copy_from_host(f_cos_host.flatten())
            
            f_sin_tensor = orchard_core.Tensor(self.backend, [curr_batch_size * (head_dim // 2)], orchard_core.DType.Float32)
            f_sin_tensor.copy_from_host(f_sin_host.flatten())
            
            # Forward
            for layer in self.layers:
                layer.forward(x_tensor, f_cos_tensor, f_sin_tensor, active_indices)
                
            # RMSNorm
            self.backend.run_rmsnorm(x_tensor, self.norm, x_tensor, self.config['rms_norm_eps'], self.config['hidden_size'], curr_batch_size)
            
            # Logits & Sampling
            logits_output = orchard_core.Tensor(self.backend, [curr_batch_size * self.config['vocab_size']], orchard_core.DType.Float32)
            
            if curr_batch_size == 1:
                self.backend.run_gemv_q4_0(self.output_w, self.output_s, x_tensor, logits_output, self.config['hidden_size'], self.config['vocab_size'])
            else:
                self.backend.run_gemm_q4_0(self.output_w, self.output_s, x_tensor, logits_output, self.config['hidden_size'], self.config['vocab_size'], curr_batch_size)
                
            logits_host = np.array(logits_output.copy_to_host(), copy=False).reshape(curr_batch_size, self.config['vocab_size'])
            
            # Sample
            next_active_indices = []
            
            for i, b_idx in enumerate(active_indices):
                logits = logits_host[i]
                next_token = self._sample(logits, temperature)
                
                generated_sequences[b_idx].append(next_token)
                current_tokens[b_idx] = next_token
                
                # Check EOS (assuming 2 is EOS for Llama 2)
                if next_token != 2: # EOS
                     next_active_indices.append(b_idx)
                     
            active_indices = next_active_indices
            
        # Decode
        results = []
        for seq in generated_sequences:
            results.append(self.tokenizer.decode(seq))
            
        return results

