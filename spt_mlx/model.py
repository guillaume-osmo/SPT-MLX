"""
SPT-MLX Model Architecture

Standalone MLX implementation of the GPT-based transformer for predicting
activity coefficients from SMILES strings.
"""

import math
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int = 100
    block_size: int = 128
    embed_size: int = 512
    num_layers: int = 6
    num_heads: int = 16
    hidden_factor: int = 4
    dropout: float = 0.0
    xT: int = 1  # Temperature/composition embedding
    mode: str = 'reg'  # 'reg' for regression


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention layer"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.embed_size % config.num_heads == 0
        
        self.key = nn.Linear(config.embed_size, config.embed_size)
        self.query = nn.Linear(config.embed_size, config.embed_size)
        self.value = nn.Linear(config.embed_size, config.embed_size)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        self.proj = nn.Linear(config.embed_size, config.embed_size)
        self.n_head = config.num_heads
        self.dropout = config.dropout

    def __call__(self, x, training=False):
        B, T, C = x.shape

        k = self.key(x)
        k = mx.reshape(k, (B, T, self.n_head, C // self.n_head))
        k = mx.transpose(k, (0, 2, 1, 3))
        
        q = self.query(x)
        q = mx.reshape(q, (B, T, self.n_head, C // self.n_head))
        q = mx.transpose(q, (0, 2, 1, 3))
        
        v = self.value(x)
        v = mx.reshape(v, (B, T, self.n_head, C // self.n_head))
        v = mx.transpose(v, (0, 2, 1, 3))

        att = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * (1.0 / math.sqrt(k.shape[-1]))
        att = nn.softmax(att, axis=-1)
        
        if training and self.dropout > 0.0:
            att = self.attn_drop(att)
            
        y = mx.matmul(att, v)
        y = mx.transpose(y, (0, 2, 1, 3))
        y = mx.reshape(y, (B, T, C))

        y = self.proj(y)
        if training and self.dropout > 0.0:
            y = self.resid_drop(y)
        return y


class MLP(nn.Module):
    """MLP block for transformer"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.embed_size, config.hidden_factor * config.embed_size)
        self.linear2 = nn.Linear(config.hidden_factor * config.embed_size, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout

    def __call__(self, x, training=False):
        x = self.linear1(x)
        x = nn.gelu(x)
        x = self.linear2(x)
        if training and self.dropout_p > 0.0:
            x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer block"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_size)
        self.ln2 = nn.LayerNorm(config.embed_size)
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def __call__(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x


class GPT(nn.Module):
    """Full GPT model for activity coefficient prediction"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.xT = config.xT
        self.regression = config.mode == 'reg'
        
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_size)
        self.pos_emb = mx.zeros((1, config.block_size + config.xT, config.embed_size))
        self.drop = nn.Dropout(config.dropout)
        self.xT_lin = nn.Linear(2, config.embed_size)
        self.blocks = [Block(config) for _ in range(config.num_layers)]
        self.ln_f = nn.LayerNorm(config.embed_size)
        self.decoder = nn.Linear(config.embed_size, config.embed_size)
        
        if config.mode == 'reg':
            self.head = nn.Linear(config.embed_size, 1)
        
        self.block_size = config.block_size

    def __call__(self, idx, xT, training=False):
        b, t = idx.shape
        
        xT = mx.expand_dims(xT, 1)
        xT_proj = self.xT_lin(xT)

        token_embeddings = self.tok_emb(idx)
        
        if self.xT > 0:
            position_embeddings = self.pos_emb[:, :t + 1, :]
            token_embeddings = mx.concatenate([token_embeddings, xT_proj], axis=1)
            x = token_embeddings + position_embeddings
        else:
            position_embeddings = self.pos_emb[:, :t, :]
            xT_embedding = mx.broadcast_to(xT_proj, (b, token_embeddings.shape[1], token_embeddings.shape[2]))
            x = token_embeddings + position_embeddings + xT_embedding
        
        if training and self.config.dropout > 0.0:
            x = self.drop(x)
        
        for block in self.blocks:
            x = block(x, training=training)
        
        x = self.ln_f(x)
        x = mx.max(x, axis=1)
        x = self.decoder(x)
        x = nn.relu(x)

        if self.config.mode == 'reg':
            logits = self.head(x)
        else:
            logits = x

        logits = mx.squeeze(logits)
        return logits
