from typing import Optional

import timm
import torch
import torch.nn as nn


def _causal_mask(seq_len: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Causal self-attention mask with -inf above the diagonal."""
    if seq_len <= 0:
        raise ValueError(f'seq_len must be positive, got {seq_len}')
    mask = torch.full((seq_len, seq_len), float('-inf'), device=device, dtype=dtype)
    return torch.triu(mask, diagonal=1)


def _token_dropout(x: torch.Tensor, p: float, *, training: bool) -> torch.Tensor:
    """Drop entire tokens (rows) in a (B, N, D) tensor with probability p.

    This is different from standard nn.Dropout, which drops individual scalar activations.
    """
    if p <= 0.0 or not training:
        return x
    if p >= 1.0:
        raise ValueError(f'token_dropout must be < 1.0, got {p}')
    if x.ndim != 3:
        raise ValueError(f'Expected token tensor of shape (B, N, D), got {tuple(x.shape)}')

    keep_prob = 1.0 - p
    mask = (torch.rand((x.shape[0], x.shape[1], 1), device=x.device) < keep_prob).to(dtype=x.dtype)
    return x * mask / keep_prob


class FrozenTimmViT(nn.Module):
    """Frozen timm ViT encoder that returns patch tokens (drops CLS)."""

    def __init__(
        self,
        encoder_name: str,
        *,
        pretrained: bool = True,
        drop_cls_token: bool = True,
        trainable: bool = False,
    ):
        super().__init__()
        # num_classes=0: no final linear layer
        # global_pool='': output full sequence (B, N, Dim), not just CLS token
        self.model = timm.create_model(encoder_name, pretrained=pretrained, num_classes=0, global_pool='')
        self.trainable = trainable
        if not self.trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.num_features = self.model.num_features
        self.drop_cls_token = drop_cls_token

    def train(self, mode: bool = True):
        if self.trainable:
            return super().train(mode)
        # Keep encoder deterministic even if the parent module is set to train().
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.trainable:
            features = self.model(images)
        else:
            with torch.no_grad():
                features = self.model(images)
        if features.ndim != 3:
            raise ValueError(f'Expected encoder output of shape (B, N, D), got {tuple(features.shape)}')
        if self.drop_cls_token:
            if features.shape[1] < 2:
                raise ValueError(f'Expected CLS + patch tokens, got N={features.shape[1]}')
            features = features[:, 1:, :]
        return features


class DecoderImageCaptioner(nn.Module):
    def __init__(
        self,
        encoder_name: str = 'vit_base_patch16_224',
        seq_len: int = 20,
        vocab_size: int = 5000,
        n_heads: int = 8,
        d_model: int = 512,
        n_layers: int = 6,
        dropout: float = 0.1,
        token_dropout: float = 0.0,
        train_backbone: bool = False,
        enc_proj_rank: int = 0,
    ):
        super().__init__()
        self.encoder = FrozenTimmViT(
            encoder_name, pretrained=True, drop_cls_token=True, trainable=train_backbone
        )
        enc_dim = self.encoder.num_features
        self.max_seq_len = seq_len
        self.enc_proj_rank = enc_proj_rank
        # image feature dimension to decoder dimension
        if enc_proj_rank and 0 < enc_proj_rank < min(enc_dim, d_model):
            self.enc_to_dec_proj = nn.Sequential(
                nn.Linear(enc_dim, enc_proj_rank, bias=False),
                nn.Linear(enc_proj_rank, d_model, bias=True),
            )
        else:
            self.enc_to_dec_proj = nn.Linear(enc_dim, d_model)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.token_dropout = token_dropout
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            activation='gelu',
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, vocab_size)

    def _encode_memory(self, images: torch.Tensor) -> torch.Tensor:
        patch_features = self.encoder(images)  # (B, N_patches, enc_dim)
        memory = self.enc_to_dec_proj(patch_features)  # (B, N_patches, d_model)
        memory = _token_dropout(memory, self.token_dropout, training=self.training)
        return self.dropout(memory)

    def _decode(self, memory: torch.Tensor, input_tokens: torch.Tensor) -> torch.Tensor:
        if input_tokens.ndim != 2:
            raise ValueError(f'Expected input_tokens of shape (B, T), got {tuple(input_tokens.shape)}')
        if input_tokens.dtype != torch.long:
            raise ValueError(f'Expected input_tokens dtype torch.long, got {input_tokens.dtype}')
        seq_len = input_tokens.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(f'input_tokens length {seq_len} exceeds max_seq_len={self.max_seq_len}')

        tgt_emb = self.text_embedding(input_tokens)
        tgt_emb = self.dropout(tgt_emb + self.pos_embedding[:, :seq_len, :])

        tgt_mask = _causal_mask(seq_len, device=memory.device, dtype=tgt_emb.dtype)
        output = self.decoder(tgt=tgt_emb, memory=memory, tgt_mask=tgt_mask)
        return self.readout(self.dropout(self.layer_norm(output)))

    def forward(self, images: torch.Tensor, target_tokens: torch.Tensor, *, start_token: int = 0) -> torch.Tensor:
        """
        images: (Batch, 3, H, W)
        target_tokens: (Batch, Seq_Len) tokens to predict
        """
        if target_tokens.ndim != 2:
            raise ValueError(f'Expected target_tokens of shape (B, T), got {tuple(target_tokens.shape)}')
        if target_tokens.dtype != torch.long:
            raise ValueError(f'Expected target_tokens dtype torch.long, got {target_tokens.dtype}')
        if target_tokens.shape[1] > self.max_seq_len:
            raise ValueError(f'target_tokens length {target_tokens.shape[1]} exceeds max_seq_len={self.max_seq_len}')

        batch_size = target_tokens.shape[0]
        decoder_input = torch.cat(
            [
                torch.full((batch_size, 1), start_token, device=target_tokens.device, dtype=torch.long),
                target_tokens[:, :-1],
            ],
            dim=1,
        )
        memory = self._encode_memory(images)
        return self._decode(memory, decoder_input)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        end_token: int,
        *,
        start_token: int = 0,
        max_length: Optional[int] = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Greedy autoregressive decoding (temperature=0) from a fixed start token."""
        if temperature != 0.0:
            raise ValueError('Only temperature=0.0 (greedy) is supported.')
        if max_length is None:
            max_length = self.max_seq_len
        if max_length < 0:
            raise ValueError(f'max_length must be >= 0, got {max_length}')
        if max_length > self.max_seq_len:
            raise ValueError(f'max_length {max_length} exceeds max_seq_len={self.max_seq_len}')

        batch_size = images.shape[0]
        device = images.device
        memory = self._encode_memory(images)

        input_tokens = torch.full((batch_size, 1), start_token, device=device, dtype=torch.long)
        generated = []

        for _ in range(max_length):
            logits = self._decode(memory, input_tokens)
            next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
            generated.append(next_tokens)

            input_tokens = torch.cat([input_tokens, next_tokens.unsqueeze(1)], dim=1)
            if torch.all(next_tokens == end_token):
                break

        if not generated:
            return input_tokens.new_empty((batch_size, 0))
        return torch.stack(generated, dim=1)
