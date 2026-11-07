"""Tests for the scratch encoder-decoder: shapes, masking, initialization, persistence."""

import pytest
import torch
import torch.nn.functional as F

from neuro_symbolic.model import (
    EncoderDecoderModel,
    ModelConfig,
    ModelError,
    additive_padding_mask,
    apply_initialization,
    causal_mask,
    initialization_report,
    load_model,
    save_model,
)


def small_config(**overrides):
    values = dict(
        vocabulary_size=97,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        attention_heads=4,
        feed_forward_size=64,
        dropout=0.1,
        max_encoder_positions=32,
        max_decoder_positions=32,
    )
    values.update(overrides)
    return ModelConfig(**values)


@pytest.fixture(scope="module")
def trained_tiny():
    return apply_initialization(EncoderDecoderModel(small_config()), seed=5).eval()


def encoder_inputs(batch: int = 3, length: int = 7):
    torch.manual_seed(0)
    ids = torch.randint(1, 90, (batch, length))
    ids[:, -2:] = 0
    return ids, ids != 0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
def test_configuration_rejects_impossible_geometry():
    with pytest.raises(ModelError):
        ModelConfig(vocabulary_size=1)
    with pytest.raises(ModelError):
        ModelConfig(vocabulary_size=97, d_model=12, attention_heads=5)
    with pytest.raises(ModelError):
        ModelConfig(vocabulary_size=97, encoder_layers=0)
    with pytest.raises(ModelError):
        ModelConfig(vocabulary_size=97, decoder_layers=0)
    with pytest.raises(ModelError):
        ModelConfig(vocabulary_size=97, dropout=1.0)
    with pytest.raises(ModelError):
        ModelConfig(vocabulary_size=97, max_decoder_positions=0)


def test_configuration_round_trips_exactly():
    config = small_config()
    assert ModelConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ModelError):
        ModelConfig.from_dict(dict(config.to_dict(), mystery=1))
    with pytest.raises(ModelError):
        ModelConfig.from_dict({"vocabulary_size": 97})


# ---------------------------------------------------------------------------
# Forward pass and attention masks
# ---------------------------------------------------------------------------
def test_forward_shapes_and_finiteness(trained_tiny):
    ids, mask = encoder_inputs()
    logits = trained_tiny(ids, mask, torch.randint(1, 90, (3, 5)))
    assert logits.shape == (3, 5, 97)
    assert bool(torch.isfinite(logits).all())


def test_positions_beyond_the_budget_are_refused(trained_tiny):
    ids, mask = encoder_inputs()
    with pytest.raises(ModelError):
        trained_tiny(ids, mask, torch.randint(1, 90, (3, 40)))
    with pytest.raises(ModelError):
        trained_tiny(torch.randint(1, 90, (3, 40)), torch.ones(3, 40, dtype=torch.bool), torch.randint(1, 90, (3, 5)))


def test_causal_attention_never_looks_to_the_right(trained_tiny):
    ids, mask = encoder_inputs()
    first = torch.randint(1, 90, (3, 4))
    extra = torch.randint(1, 90, (3, 3))
    with torch.no_grad():
        short = trained_tiny(ids, mask, first)
        long = trained_tiny(ids, mask, torch.cat([first, extra], dim=1))
    assert torch.allclose(short, long[:, :4, :], atol=1e-6)
    with torch.no_grad():
        unmasked = trained_tiny(ids, mask, first, causal=False)
    assert not torch.allclose(short, unmasked, atol=1e-5)


def test_causal_and_padding_masks_have_the_expected_geometry():
    causal = causal_mask(5, torch.device("cpu"))
    assert causal.shape == (1, 1, 5, 5)
    assert float(causal[0, 0, 0, 1]) == float("-inf")
    assert float(causal[0, 0, 4, 4]) == 0.0
    assert float(causal[0, 0, 4, 0]) == 0.0

    ids, mask = encoder_inputs()
    additive = additive_padding_mask(mask)
    assert additive.shape == (3, 1, 1, 7)
    assert float(additive[0, 0, 0, 5]) == float("-inf")
    assert float(additive[0, 0, 0, 0]) == 0.0


def test_padding_on_the_encoder_side_changes_nothing(trained_tiny):
    ids, mask = encoder_inputs()
    decoder = torch.randint(1, 90, (3, 5))
    with torch.no_grad():
        plain = trained_tiny(ids, mask, decoder)
        padded_ids = torch.cat([ids, torch.zeros(3, 2, dtype=torch.long)], dim=1)
        padded_mask = torch.cat([mask, torch.zeros(3, 2, dtype=torch.bool)], dim=1)
        padded = trained_tiny(padded_ids, padded_mask, decoder)
    assert torch.allclose(plain, padded, atol=1e-6)


def test_the_embedding_of_the_padding_token_stays_zero(trained_tiny):
    assert bool((trained_tiny.encoder_embeddings.tokens.weight[0] == 0).all())
    assert bool((trained_tiny.decoder_embeddings.tokens.weight[0] == 0).all())


def test_output_projection_is_tied_to_the_decoder_table_and_encoder_stays_separate():
    model = apply_initialization(EncoderDecoderModel(small_config()), seed=1)
    assert model.output_projection.weight is model.decoder_embeddings.tokens.weight
    assert model.output_projection.weight is not model.encoder_embeddings.tokens.weight
    untied = apply_initialization(EncoderDecoderModel(small_config(tie_word_embeddings=False)), seed=1)
    assert untied.output_projection.weight is not untied.decoder_embeddings.tokens.weight


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
def test_initialization_is_reproducible_per_seed_and_changes_across_seeds():
    a = apply_initialization(EncoderDecoderModel(small_config()), seed=11)
    b = apply_initialization(EncoderDecoderModel(small_config()), seed=11)
    c = apply_initialization(EncoderDecoderModel(small_config()), seed=12)
    left = a.encoder[0].self_attention.query.weight
    assert torch.equal(left, b.encoder[0].self_attention.query.weight)
    assert torch.equal(a.decoder_embeddings.tokens.weight, b.decoder_embeddings.tokens.weight)
    assert not torch.equal(left, c.encoder[0].self_attention.query.weight)


def test_layer_norms_start_neutral_and_no_weight_is_corrupted():
    model = apply_initialization(EncoderDecoderModel(small_config()), seed=3)
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            assert bool((module.weight == 1).all())
            assert bool((module.bias == 0).all())
    report = initialization_report(model)
    assert report["non_finite_parameters"] == 0
    assert 0.01 < report["standard_deviation"] < 0.3


def test_feed_forward_matrices_use_the_relu_side_scheme():
    model = apply_initialization(EncoderDecoderModel(small_config()), seed=7)
    block = model.encoder[0]
    attention = float(block.self_attention.query.weight.detach().abs().max())
    inner = float(block.feed_forward.inner.weight.detach().abs().max())
    outer = float(block.feed_forward.outer.weight.detach().abs().max())
    fan_in = small_config().d_model
    xavier_bound = (6.0 / (fan_in + fan_in)) ** 0.5
    he_bound = (6.0 / fan_in) ** 0.5
    assert attention <= xavier_bound + 1e-6
    assert outer <= xavier_bound + 1e-6
    assert inner <= he_bound + 1e-6
    assert inner > xavier_bound
    assert inner / attention > 1.2


def test_biases_are_small_and_gradients_reach_every_tensor():
    model = apply_initialization(EncoderDecoderModel(small_config()), seed=2)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            bound = 1.0 / (module.weight.shape[1] ** 0.5)
            assert float(module.bias.detach().abs().max()) <= bound + 1e-9, name
    ids, mask = encoder_inputs()
    logits = model(ids, mask, torch.randint(1, 90, (3, 5)))
    F.cross_entropy(logits[:, :-1].reshape(-1, 97), torch.randint(1, 90, (3, 4)).reshape(-1)).backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert not missing
    assert all(float(parameter.grad.abs().sum()) > 0 for parameter in model.parameters())


# ---------------------------------------------------------------------------
# Parameter accounting
# ---------------------------------------------------------------------------
def test_parameter_breakdown_matches_the_total():
    config = small_config()
    model = apply_initialization(EncoderDecoderModel(config), seed=0)
    breakdown = model.parameter_breakdown()
    parts = (
        breakdown["encoder_blocks"]
        + breakdown["decoder_blocks"]
        + breakdown["token_tables"]
        + breakdown["position_tables"]
        + breakdown["final_norms"]
        + breakdown["tied_output_projection"]
    )
    assert parts == breakdown["total"] == sum(p.numel() for p in model.parameters())
    assert breakdown["trainable"] == breakdown["total"]
    per_block = (
        4 * config.d_model * config.d_model
        + 4 * config.d_model
        + 2 * config.d_model * config.feed_forward_size
        + config.feed_forward_size
        + config.d_model
        + 4 * config.d_model
    )
    assert breakdown["encoder_blocks"] == config.encoder_layers * per_block


def test_tying_shrinks_the_counted_weights():
    tied = apply_initialization(EncoderDecoderModel(small_config()), seed=0).parameter_breakdown()
    untied = apply_initialization(EncoderDecoderModel(small_config(tie_word_embeddings=False)), seed=0).parameter_breakdown()
    assert untied["total"] - tied["total"] == 97 * small_config().d_model


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_saved_weights_reload_to_the_same_function(tmp_path):
    model = apply_initialization(EncoderDecoderModel(small_config()), seed=9)
    ids, mask = encoder_inputs()
    targets = torch.randint(1, 90, (3, 5))
    model.eval()
    with torch.no_grad():
        before = model(ids, mask, targets)
    path = str(tmp_path / "weights.pt")
    save_model(model, path, metadata={"epoch": 3, "note": "round trip"})
    restored = load_model(path)
    with torch.no_grad():
        after = restored(ids, mask, targets)
    assert torch.equal(before, after)
    assert restored.training is False


def test_a_foreign_archive_is_refused(tmp_path):
    path = tmp_path / "other.pt"
    torch.save({"format": "someone-elses-checkpoint", "config": {}}, path)
    with pytest.raises(ModelError):
        load_model(str(path))
