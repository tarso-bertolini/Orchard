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
        
        # KV Cache (CPU numpy arrays, uploaded on demand)
        # k_cache: List of (HeadDim, SeqLen) per head? 
        # Easier: (Heads, HeadDim, SeqLen)
        self.k_cache_np = np.zeros((self.heads, self.head_dim, 0), dtype=np.float32)
        self.v_cache_np = np.zeros((self.heads, 0, self.head_dim), dtype=np.float32)

    def forward(self, x, freqs_cos, freqs_sin, start_pos):
        # x: [Hidden] (Tensor) - Input embedding/state
        # freqs_cos/sin: [1, HeadDim/2] (Tensor) - RoPE for current position
        
        # 1. RMSNorm
        # We need a temp tensor for output of norm
        x_norm = orchard_core.Tensor(self.backend, [self.hidden_dim], orchard_core.DType.Float32)
        self.backend.run_rmsnorm(x, self.attn_norm, x_norm, self.config['rms_norm_eps'], self.hidden_dim, 1)
        
        # 2. Q, K, V Projections
        # x_norm is (Hidden). W is (Hidden, Hidden).
        # run_gemv_q4_0: weights(N, K/2), scales(N, K/32), input(K), output(N)
        # Here N=Hidden, K=Hidden.
        
        q = orchard_core.Tensor(self.backend, [self.hidden_dim], orchard_core.DType.Float32)
        k = orchard_core.Tensor(self.backend, [self.hidden_dim], orchard_core.DType.Float32)
        v = orchard_core.Tensor(self.backend, [self.hidden_dim], orchard_core.DType.Float32)
        
        self.backend.run_gemv_q4_0(self.wq_w, self.wq_s, x_norm, q, self.hidden_dim, self.hidden_dim)
        self.backend.run_gemv_q4_0(self.wk_w, self.wk_s, x_norm, k, self.hidden_dim, self.hidden_dim)
        self.backend.run_gemv_q4_0(self.wv_w, self.wv_s, x_norm, v, self.hidden_dim, self.hidden_dim)
        
        # 3. RoPE
        # In-place on q and k
        # run_rope(input, cos, sin, output, head_dim, num_heads, seq_len)
        # seq_len=1 for generation
        self.backend.run_rope(q, freqs_cos, freqs_sin, q, self.head_dim, self.heads, 1)
        self.backend.run_rope(k, freqs_cos, freqs_sin, k, self.head_dim, self.heads, 1)
        
        # 4. Update KV Cache & Attention
        # Download Q, K, V to CPU to handle multi-head splitting and caching
        # This is the bottleneck for v0.1 but necessary without complex kernels
        q_cpu = np.array(q.copy_to_host(), copy=False).reshape(self.heads, self.head_dim)
        k_cpu = np.array(k.copy_to_host(), copy=False).reshape(self.heads, self.head_dim)
        v_cpu = np.array(v.copy_to_host(), copy=False).reshape(self.heads, self.head_dim)
        
        # Append to cache
        # k_cpu: (Heads, HeadDim). We want to append to (Heads, HeadDim, Seq)
        self.k_cache_np = np.dstack([self.k_cache_np, k_cpu[:, :, None]])
        # v_cpu: (Heads, HeadDim). We want to append to (Heads, Seq, HeadDim)
        self.v_cache_np = np.hstack([self.v_cache_np, v_cpu[:, None, :]])
        
        seq_len = self.k_cache_np.shape[2]
        
        # Attention Calculation (per head)
        # We can do this on CPU for now since we are already here, 
        # OR upload per-head buffers to GPU.
        # Given the overhead of 32 uploads/downloads, CPU might be faster for short sequences.
        # Let's stick to CPU for the attention mechanism in v0.1 for simplicity and reliability.
        # Metal is great for the big MatMuls (Projections, FFN). Attention on small vectors is fine on CPU.
        
        output_cpu = np.zeros((self.heads, self.head_dim), dtype=np.float32)
        
        scale = 1.0 / np.sqrt(self.head_dim)
        
        for h in range(self.heads):
            # Q: (HeadDim)
            # K_cache: (HeadDim, Seq)
            # Score: (Seq)
            qs = q_cpu[h]
            ks = self.k_cache_np[h] # (HeadDim, Seq)
            vs = self.v_cache_np[h] # (Seq, HeadDim)
            
            score = np.matmul(qs, ks) * scale
            
            # Softmax
            score_max = np.max(score)
            exp_score = np.exp(score - score_max)
            softmax = exp_score / np.sum(exp_score)
            
            # Output: (HeadDim) = Softmax(Seq) @ V(Seq, HeadDim)
            output_cpu[h] = np.matmul(softmax, vs)
            
        # Flatten output
        output_flat = output_cpu.flatten()
        
        # Upload back to GPU for Output Projection
        attn_output = orchard_core.Tensor(self.backend, [self.hidden_dim], orchard_core.DType.Float32)
        attn_output.copy_from_host(output_flat)
        
        # 5. Output Projection
        # run_gemv_q4_0: weights(N, K/2), scales(N, K/32), input(K), output(N)
        # Here N=Hidden, K=Hidden
        attn_proj = orchard_core.Tensor(self.backend, [self.hidden_dim], orchard_core.DType.Float32)
        self.backend.run_gemv_q4_0(self.wo_w, self.wo_s, attn_output, attn_proj, self.hidden_dim, self.hidden_dim)
        
        # 6. Residual Add
        # x = x + attn_proj
        self.backend.run_add(x, attn_proj, x, self.hidden_dim)
        
        # 7. FFN
        # Norm
        ffn_input = orchard_core.Tensor(self.backend, [self.hidden_dim], orchard_core.DType.Float32)
        self.backend.run_rmsnorm(x, self.ffn_norm, ffn_input, self.config['rms_norm_eps'], self.hidden_dim, 1)
        
        # Gate & Up
        # Gate: w1 (Inter, Hidden)
        gate = orchard_core.Tensor(self.backend, [self.intermediate_dim], orchard_core.DType.Float32)
        self.backend.run_gemv_q4_0(self.w1_w, self.w1_s, ffn_input, gate, self.hidden_dim, self.intermediate_dim)
        
        # Up: w3 (Inter, Hidden)
        up = orchard_core.Tensor(self.backend, [self.intermediate_dim], orchard_core.DType.Float32)
        self.backend.run_gemv_q4_0(self.w3_w, self.w3_s, ffn_input, up, self.hidden_dim, self.intermediate_dim)
        
        # Silu on Gate
        self.backend.run_silu(gate, gate, self.intermediate_dim)
        
        # Multiply Gate * Up
        self.backend.run_mul(gate, up, gate, self.intermediate_dim)
        
        # Down: w2 (Hidden, Inter)
        ffn_output = orchard_core.Tensor(self.backend, [self.hidden_dim], orchard_core.DType.Float32)
        self.backend.run_gemv_q4_0(self.w2_w, self.w2_s, gate, ffn_output, self.intermediate_dim, self.hidden_dim)
        
        # Residual Add
        self.backend.run_add(x, ffn_output, x, self.hidden_dim)
        
        return x

class Llama:
    def __init__(self, model_path: str):
        self.backend = orchard_core.MetalBackend()
        self.backend.initialize()
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
        
    def generate(self, prompt: str, max_tokens: int = 50, temperature: float = 0.7):
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
            
        # Encode
        ids = self.tokenizer.encode(prompt).ids
        
        # Generation Loop
        for i in range(max_tokens):
            # Prepare input tensor
            # For the first step, we process the prompt.
            # For subsequent steps, we process the last token.
            # NOTE: For v0.1, we process token-by-token even for the prompt (slow fill)
            # to avoid implementing batched processing logic right now.
            
            if i == 0:
                # Process prompt
                for t_idx, token_id in enumerate(ids):
                    next_token = self._forward_one(token_id, t_idx)
            else:
                # Process generated token
                next_token = self._forward_one(next_token, len(ids) + i - 1)
                
            # Sample
            # next_token is actually the logits or the token?
            # _forward_one returns the logits for the next token
            logits = next_token
            
            # Greedy / Temperature
            if temperature == 0:
                token_id = np.argmax(logits)
            else:
                # Softmax
                probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
                token_id = np.random.choice(len(probs), p=probs)
                
            print(self.tokenizer.decode([token_id]), end="", flush=True)
            ids.append(token_id)
            next_token = token_id # For next loop
            
        print("") # Newline
        
    def _forward_one(self, token_id: int, pos: int):
        # 1. Embedding
        # We need to look up the embedding vector.
        # Since we don't have an embedding kernel that takes a scalar index easily from Python without wrapping,
        # we can just slice the embedding matrix on CPU if it's not quantized, or use the kernel.
        # Actually, the embedding weights are likely on GPU if we loaded them?
        # Wait, loader.py doesn't load embeddings yet!
        # Let's check loader.py again.
        
        # We need to ensure embeddings are loaded.
        # Assuming self.embed_tokens exists (we need to add it to loader)
        
        # For now, let's assume we have self.embed_tokens_w (quantized) or fp32.
        # If it's quantized, we can't easily slice it on CPU.
        # We should use the embedding kernel: run_embedding(input_ids, weights, output, num_tokens, hidden_dim)
        
        input_ids_t = orchard_core.Tensor(self.backend, [1], orchard_core.DType.Int32) # Wait, Int32 support?
        # My Tensor only supports Float32, Float16, Int8.
        # I need to add Int32 support to Tensor or just pass a pointer.
        # The kernel takes `int* input_ids`.
        # I can pass a Float32 tensor and cast it in the kernel? No, that's unsafe.
        
        # Workaround: Pass input_ids as host pointer?
        # Or just implement `run_embedding_scalar` in bindings?
        
        # Let's assume for now we can just get the vector.
        # If embeddings are FP32 on CPU (common for embeddings), it's easy.
        # If they are on GPU, we need a kernel.
        
        # Let's assume embeddings are on CPU for v0.1 to save VRAM and complexity.
        # We will update loader to keep embeddings on CPU.
        
        x_emb = self.embed_tokens[token_id] # Numpy array
        
        # Upload to GPU
        x = orchard_core.Tensor(self.backend, [self.config['hidden_size']], orchard_core.DType.Float32)
        x.copy_from_host(x_emb)
        
        # RoPE freqs for this position
        # freqs_cos: (Seq, HeadDim/2)
        cos = self.freqs_cos[pos]
        sin = self.freqs_sin[pos]
        
        # Upload RoPE
        # We need to repeat them for all heads?
        # The kernel takes (HeadDim/2). It broadcasts across heads?
        # run_rope implementation:
        # device const float* freqs_cos [[buffer(1)]], ...
        # It reads `freqs_cos[i % head_dim]`.
        # So we just need one copy of cos/sin for the head dimension.
        
        t_cos = orchard_core.Tensor(self.backend, [self.config['hidden_size'] // self.config['num_attention_heads'] // 2], orchard_core.DType.Float32)
        t_sin = orchard_core.Tensor(self.backend, [self.config['hidden_size'] // self.config['num_attention_heads'] // 2], orchard_core.DType.Float32)
        t_cos.copy_from_host(cos)
        t_sin.copy_from_host(sin)
        
        # Run Layers
        for layer in self.layers:
            x = layer.forward(x, t_cos, t_sin, pos)
            
        # Final Norm
        x_final = orchard_core.Tensor(self.backend, [self.config['hidden_size']], orchard_core.DType.Float32)
        self.backend.run_rmsnorm(x, self.norm, x_final, self.config['rms_norm_eps'], self.config['hidden_size'], 1)
        
        # Output Head
        # self.output_w (Quantized)
        # vocab_size = self.config['vocab_size']
        logits = orchard_core.Tensor(self.backend, [self.config['vocab_size']], orchard_core.DType.Float32)
        self.backend.run_gemv_q4_0(self.output_w, self.output_s, x_final, logits, self.config['hidden_size'], self.config['vocab_size'])
        
        return np.array(logits.copy_to_host(), copy=False)

