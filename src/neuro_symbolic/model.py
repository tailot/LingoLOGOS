"""Encoder-decoder transformer of the pure model, built from empty tensors.

Nothing in this module is read from a checkpoint of somebody else: every weight
comes from a stochastic draw that :func:`apply_initialization` performs, and the
vocabulary is the one learned by :mod:`neuro_symbolic.tokenizer`. The architecture
follows ``doc/PURE_MODEL_ROADMAP.md``: six encoder layers and six decoder layers,
eight attention heads, ``d_model`` 512, feed-forward width 2048, dropout 0.1,
learned absolute positions sized from the measured sequence lengths.

The network is deliberately unaware of the algebra. Legality of the output is
enforced outside it, by the decoding mask of :mod:`neuro_symbolic.masking`, so this
module only produces scores over the whole vocabulary and lets the caller decide
which entries are reachable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

FORMAT_NAME = "lingologos-transformer-1"
INIT_STDDEV = 0.02


class ModelError(Exception):
    """Raised for an inconsistent configuration or an out-of-range sequence."""


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters of one model instance, serializable exactly as they stand."""

    vocabulary_size: int
    d_model: int = 512
    encoder_layers: int = 6
    decoder_layers: int = 6
    attention_heads: int = 8
    feed_forward_size: int = 2048
    dropout: float = 0.1
    max_encoder_positions: int = 96
    max_decoder_positions: int = 96
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocabulary_size < 2:
            raise ModelError("vocabulary_size must be at least 2")
        if self.d_model < 1 or self.feed_forward_size < 1:
            raise ModelError("model width must be positive")
        if self.attention_heads < 1 or self.d_model % self.attention_heads:
            raise ModelError("d_model must be divisible by attention_heads")
        if self.encoder_layers < 1 or self.decoder_layers < 1:
            raise ModelError("the network needs at least one layer on each side")
        if not 0.0 <= self.dropout < 1.0:
            raise ModelError("dropout must live in the interval [0, 1)")
        if self.max_encoder_positions < 1 or self.max_decoder_positions < 1:
            raise ModelError("position budgets must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelConfig":
        fields = set(cls.__annotations__)
        unknown = sorted(set(payload) - fields)
        if unknown:
            raise ModelError("unknown configuration keys: " + ", ".join(unknown))
        missing = sorted(fields - set(payload))
        if missing:
            raise ModelError("missing configuration keys: " + ", ".join(missing))
        return cls(**{name: payload[name] for name in fields})


# ---------------------------------------------------------------------------
# Layers. They are written here rather than borrowed from a library container so
# that the parameter count and the initialization of every matrix are explicit.
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """Multi-head attention: self-attention over one stream, or over encoder memory."""

    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ModelError("d_model must be divisible by the head count")
        self.d_model = int(d_model)
        self.heads = int(heads)
        self.head_width = self.d_model // self.heads
        self.dropout = float(dropout)
        self.query = nn.Linear(self.d_model, self.d_model)
        self.key = nn.Linear(self.d_model, self.d_model)
        self.value = nn.Linear(self.d_model, self.d_model)
        self.output = nn.Linear(self.d_model, self.d_model)

    def _split(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(tensor.shape[0], tensor.shape[1], self.heads, self.head_width).transpose(1, 2)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query = self._split(self.query(queries))
        key = self._split(self.key(keys))
        value = self._split(self.value(keys))
        context = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(queries.shape[0], queries.shape[1], self.d_model)
        return self.output(context)


class FeedForward(nn.Module):
    """Two linear maps with a ReLU between them, the pair the specification sizes at 2048."""

    def __init__(self, d_model: int, inner: int) -> None:
        super().__init__()
        self.inner = nn.Linear(d_model, int(inner))
        self.outer = nn.Linear(int(inner), d_model)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.outer(F.relu(self.inner(inputs)))


class TransformerBlock(nn.Module):
    """Pre-normalization residual block, with a cross-attention sublayer when decoding."""

    def __init__(
        self,
        d_model: int,
        heads: int,
        feed_forward_size: int,
        dropout: float,
        cross_attention: bool = False,
    ) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.self_norm = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model, heads, dropout) if cross_attention else None
        self.cross_norm = nn.LayerNorm(d_model) if cross_attention else None
        self.feed_forward = FeedForward(d_model, feed_forward_size)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        self_mask: Optional[torch.Tensor],
        memory: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden = self.self_norm(inputs)
        inputs = inputs + self.dropout(self.self_attention(hidden, hidden, self_mask))
        if self.cross_attention is not None and memory is not None:
            hidden = self.cross_norm(inputs)
            inputs = inputs + self.dropout(self.cross_attention(hidden, memory, memory_mask))
        hidden = self.feed_forward_norm(inputs)
        return inputs + self.dropout(self.feed_forward(hidden))


class TokenAndPositionEmbeddings(nn.Module):
    """Learned token table plus learned absolute positions, tokens scaled by sqrt(d)."""

    def __init__(self, vocabulary_size: int, d_model: int, max_positions: int, dropout: float, padding_id: int) -> None:
        super().__init__()
        self.tokens = nn.Embedding(int(vocabulary_size), int(d_model), padding_idx=int(padding_id))
        self.positions = nn.Embedding(int(max_positions), int(d_model))
        self.padding_id = int(padding_id)
        self.max_positions = int(max_positions)
        self.scale = math.sqrt(int(d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length > self.max_positions:
            raise ModelError(
                "sequence of " + str(length) + " tokens exceeds the position budget " + str(self.max_positions)
            )
        positions = torch.arange(length, device=ids.device, dtype=torch.long)
        return self.dropout(self.tokens(ids) * self.scale + self.positions(positions))


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
class EncoderDecoderModel(nn.Module):
    """Transformer encoder-decoder producing logits over the learned vocabulary."""

    def __init__(self, config: ModelConfig, padding_id: int = 0) -> None:
        super().__init__()
        self.config = config
        self.padding_id = int(padding_id)
        self.encoder_embeddings = TokenAndPositionEmbeddings(
            config.vocabulary_size, config.d_model, config.max_encoder_positions, config.dropout, padding_id
        )
        self.decoder_embeddings = TokenAndPositionEmbeddings(
            config.vocabulary_size, config.d_model, config.max_decoder_positions, config.dropout, padding_id
        )
        self.encoder = nn.ModuleList(
            TransformerBlock(config.d_model, config.attention_heads, config.feed_forward_size, config.dropout)
            for _ in range(config.encoder_layers)
        )
        self.decoder = nn.ModuleList(
            TransformerBlock(
                config.d_model, config.attention_heads, config.feed_forward_size, config.dropout, True
            )
            for _ in range(config.decoder_layers)
        )
        self.encoder_final_norm = nn.LayerNorm(config.d_model)
        self.decoder_final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocabulary_size, bias=False)
        if config.tie_word_embeddings:
            self.output_projection.weight = self.decoder_embeddings.tokens.weight

    # -- forward passes ---------------------------------------------------
    def encode(self, input_ids: torch.Tensor, input_mask: torch.Tensor) -> torch.Tensor:
        """Memory states of the encoder, one vector per input token."""
        hidden = self.encoder_embeddings(input_ids)
        attention_mask = additive_padding_mask(input_mask)
        for block in self.encoder:
            hidden = block(hidden, attention_mask)
        return self.encoder_final_norm(hidden)

    def decode(
        self,
        decoder_ids: torch.Tensor,
        memory: torch.Tensor,
        input_mask: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        """Logits of shape ``[batch, length, vocabulary]`` for each decoder position."""
        hidden = self.decoder_embeddings(decoder_ids)
        self_mask = causal_mask(decoder_ids.shape[1], decoder_ids.device) if causal else None
        memory_mask = additive_padding_mask(input_mask)
        for block in self.decoder:
            hidden = block(hidden, self_mask, memory, memory_mask)
        hidden = self.decoder_final_norm(hidden)
        return self.output_projection(hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        input_mask: torch.Tensor,
        decoder_ids: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        return self.decode(decoder_ids, self.encode(input_ids, input_mask), input_mask, causal)

    # -- reporting --------------------------------------------------------
    def parameter_breakdown(self) -> Dict[str, int]:
        """Totals split the way the roadmap budgets them."""
        def total(modules: Iterable[nn.Module]) -> int:
            return sum(int(p.numel()) for module in modules for p in module.parameters())

        embedding_rows = self.decoder_embeddings.tokens.num_embeddings
        table = embedding_rows * self.decoder_embeddings.tokens.embedding_dim
        return {
            "encoder_blocks": total(self.encoder),
            "decoder_blocks": total(self.decoder),
            "token_tables": 2 * table,
            "position_tables": self.encoder_embeddings.positions.weight.numel()
            + self.decoder_embeddings.positions.weight.numel(),
            "final_norms": int(sum(p.numel() for p in self.encoder_final_norm.parameters()))
            + int(sum(p.numel() for p in self.decoder_final_norm.parameters())),
            "tied_output_projection": 0 if self.config.tie_word_embeddings else table,
            "total": int(sum(p.numel() for p in self.parameters())),
            "trainable": int(sum(p.numel() for p in self.parameters() if p.requires_grad)),
        }


# ---------------------------------------------------------------------------
# Masks the attention needs
# ---------------------------------------------------------------------------
def additive_padding_mask(input_mask: torch.Tensor) -> torch.Tensor:
    """Turn a ``True keeps the key`` mask of shape ``[batch, length]`` into the additive
    mask of shape ``[batch, 1, 1, length]`` that attention broadcasts over the heads."""
    mask = torch.zeros(input_mask.shape, dtype=torch.float32, device=input_mask.device)
    mask.masked_fill_(~input_mask, float("-inf"))
    return mask.unsqueeze(1).unsqueeze(1)


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Additive mask that forbids attending to a later position."""
    mask = torch.full((length, length), float("-inf"), dtype=torch.float32, device=device)
    mask.triu_(1)
    return mask.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# Initialization: the only source of weights in this repository.
# ---------------------------------------------------------------------------
def apply_initialization(model: EncoderDecoderModel, seed: int = 0) -> EncoderDecoderModel:
    """Draw every parameter from scratch, then return the same model.

    Per the specification: embedding tables and the output projection follow a
    normal distribution of standard deviation 0.02, the ReLU-side matrices of the
    feed-forward blocks get the Kaiming scheme, every other projection gets Xavier,
    layer-normalization gains stay neutral, and the padding row of each token table
    stays at zero.
    """
    torch.manual_seed(int(seed))
    relu_side: Set[int] = set()
    for block in list(model.encoder) + list(model.decoder):
        relu_side.add(id(block.feed_forward.inner))
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            with torch.no_grad():
                module.weight.fill_(1.0)
                module.bias.zero_()
        elif isinstance(module, nn.Linear):
            if id(module) in relu_side:
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5), mode="fan_in", nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                bound = 1.0 / math.sqrt(module.weight.shape[1])
                nn.init.uniform_(module.bias, -bound, bound)
    for embeddings in (model.encoder_embeddings, model.decoder_embeddings):
        nn.init.normal_(embeddings.tokens.weight, 0.0, INIT_STDDEV)
        nn.init.normal_(embeddings.positions.weight, 0.0, INIT_STDDEV)
        with torch.no_grad():
            embeddings.tokens.weight[model.padding_id].zero_()
    if model.config.tie_word_embeddings:
        model.output_projection.weight = model.decoder_embeddings.tokens.weight
    return model


def initialization_report(model: EncoderDecoderModel) -> Dict[str, Any]:
    """Summary statistics of the draw, to catch an initialization that collapsed."""
    weights = [p.detach().float() for p in model.parameters() if p.dim() >= 2]
    return {
        "matrices": len(weights),
        "standard_deviation": round(float(torch.cat([w.flatten() for w in weights]).std()), 6),
        "smallest_matrix_rows": min(int(w.shape[0]) for w in weights),
        "largest_matrix_rows": max(int(w.shape[0]) for w in weights),
        "non_finite_parameters": int(sum(int(torch.isfinite(w).logical_not().sum()) for w in weights)),
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def save_model(model: EncoderDecoderModel, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """Write configuration, weights and provenance in one torch archive."""
    payload: Dict[str, Any] = {
        "format": FORMAT_NAME,
        "config": model.config.to_dict(),
        "padding_id": model.padding_id,
        "state_dict": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, path)


def load_model(path: str, device: Optional[str] = None) -> EncoderDecoderModel:
    """Read back an archive written by :func:`save_model` and refuse anything else."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch releases without the weights_only switch
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ModelError("not a " + FORMAT_NAME + " archive: " + str(path))
    model = EncoderDecoderModel(
        ModelConfig.from_dict(payload["config"]),
        padding_id=int(payload.get("padding_id", 0)),
    )
    model.load_state_dict(payload["state_dict"])
    if device:
        model.to(device)
    model.eval()
    return model


def load_model_metadata(path: str) -> Dict[str, Any]:
    """Return the provenance block of an archive without rebuilding the model."""
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or payload.get("format") != FORMAT_NAME:
        raise ModelError("not a " + FORMAT_NAME + " archive: " + str(path))
    metadata = dict(payload.get("metadata") or {})
    metadata["config"] = payload.get("config")
    return metadata
