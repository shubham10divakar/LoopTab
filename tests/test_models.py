import pytest
import torch

from looptab.models import (
    VARIANTS,
    DenseResidualAggregator,
    FeatureTokenizer,
    LoopTab,
    build_looptab,
    count_parameters,
)

B, N_NUM, CARDS = 8, 5, [3, 7]


def _inputs():
    torch.manual_seed(0)
    x_num = torch.randn(B, N_NUM)
    x_cat = torch.stack([torch.randint(0, c + 1, (B,)) for c in CARDS], dim=1)
    return x_num, x_cat


def test_tokenizer_shape_and_per_feature_projection():
    tok = FeatureTokenizer(N_NUM, CARDS, d_model=16)
    x_num, x_cat = _inputs()
    z = tok(x_num, x_cat)
    assert z.shape == (B, 1 + N_NUM + len(CARDS), 16)
    # numerical token j is exactly x_j * W_j + b_j
    expected = x_num[:, 2:3] * tok.num_weight[2] + tok.num_bias[2]
    assert torch.allclose(z[:, 1 + 2], expected)


def test_tokenizer_categorical_tables_are_separate():
    tok = FeatureTokenizer(0, [2, 2], d_model=4)
    z = tok(None, torch.tensor([[1, 1]]))
    # same code in different columns must map to different embeddings
    assert not torch.allclose(z[0, 1], z[0, 2])


def test_aggregator_starts_uniform_and_sums():
    agg = DenseResidualAggregator(3)
    assert torch.allclose(agg.weights, torch.full((4,), 0.25))
    outs = [torch.full((2, 3, 4), float(i)) for i in range(4)]
    assert torch.allclose(agg(outs), torch.full((2, 3, 4), 1.5))


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_variants_forward_backward(variant):
    model = build_looptab(variant, N_NUM, CARDS, d_model=16, n_heads=4, d_ff=32)
    x_num, x_cat = _inputs()
    out = model(x_num, x_cat, return_aux=True)
    assert out.logits.shape == (B,)
    assert len(out.attn_maps) == model.n_loops
    assert out.attn_maps[0].shape == (B, 4, 1 + N_NUM + len(CARDS), 1 + N_NUM + len(CARDS))
    if variant == "looptab-nogate":
        assert out.gates == []
    else:
        assert len(out.gates) == model.n_loops
        assert ((out.gates[0] >= 0) & (out.gates[0] <= 1)).all()
    assert (out.alpha is None) == (variant == "looptab-nodense")
    out.logits.sum().backward()


def test_weight_tying_param_counts():
    kw = dict(d_model=32, n_heads=4, d_ff=64)
    tied = LoopTab(N_NUM, CARDS, n_loops=4, tied_weights=True, **kw)
    tied_k1 = LoopTab(N_NUM, CARDS, n_loops=1, tied_weights=True, **kw)
    untied = LoopTab(N_NUM, CARDS, n_loops=4, tied_weights=False, **kw)
    block = count_parameters(tied.blocks[0])
    # tied K=4 only adds K extra aggregator logits over K=1
    assert count_parameters(tied) == count_parameters(tied_k1) + 3
    assert count_parameters(untied) == count_parameters(tied) + 3 * block


def test_return_logits_only_by_default():
    model = build_looptab("looptab-s", N_NUM, CARDS)
    x_num, x_cat = _inputs()
    assert model(x_num, x_cat).shape == (B,)


def test_numerical_only_model():
    model = build_looptab("looptab-s", N_NUM)
    assert model(torch.randn(B, N_NUM)).shape == (B,)
