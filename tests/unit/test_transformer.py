import torch

from aqcascade.models.transformer import PollutionTransformer


def test_forward_pass_produces_expected_output_shape():
    model = PollutionTransformer(
        n_channels=15, n_horizons=2, seq_len=24, d_model=16, nhead=2, num_layers=1
    )
    x = torch.randn(8, 24, 15)
    out = model(x)
    assert out.shape == (8, 2)


def test_prediction_is_sensitive_to_the_current_and_past_steps():
    """The model should actually use the sequence, not ignore it -- a sanity
    check that perturbing either the most recent step or an earlier one
    changes the prediction (both are legitimately "the past" relative to
    the prediction target, so both are allowed to matter)."""
    torch.manual_seed(0)
    model = PollutionTransformer(
        n_channels=4, n_horizons=1, seq_len=6, d_model=8, nhead=2, num_layers=1
    )
    model.eval()

    x = torch.randn(1, 6, 4)
    with torch.no_grad():
        out_orig = model(x)

    x_perturbed_last = x.clone()
    x_perturbed_last[0, -1, :] += 100.0
    x_perturbed_first = x.clone()
    x_perturbed_first[0, 0, :] += 100.0

    with torch.no_grad():
        out_last = model(x_perturbed_last)
        out_first = model(x_perturbed_first)

    assert not torch.allclose(out_orig, out_last)
    assert not torch.allclose(out_orig, out_first)
