"""Unit tests for the GestureClassifierTCN architecture."""

import pytest

pytest.importorskip("torch", reason="torch not installed")

import torch  # noqa: E402 (after importorskip)

from airtrack.ml.tcn_model import GestureClassifierTCN  # noqa: E402


@pytest.fixture
def model() -> GestureClassifierTCN:
    return GestureClassifierTCN(feature_dim=7, hidden_channels=16, num_blocks=3)


def test_output_shape(model: GestureClassifierTCN) -> None:
    x = torch.zeros(4, 12, 7)  # batch=4, T=12 frames, 7 features
    logits = model(x)
    assert logits.shape == (4, GestureClassifierTCN.NUM_CLASSES)


def test_output_dtype(model: GestureClassifierTCN) -> None:
    x = torch.zeros(1, 12, 7)
    logits = model(x)
    assert logits.dtype == torch.float32


def test_single_frame_input(model: GestureClassifierTCN) -> None:
    """Model must not crash on T=1 (edge case at startup)."""
    x = torch.zeros(1, 1, 7)
    logits = model(x)
    assert logits.shape == (1, 2)


def test_gradient_flows(model: GestureClassifierTCN) -> None:
    x = torch.randn(2, 12, 7)
    labels = torch.tensor([0, 1])
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"


def test_torchscript_export(model: GestureClassifierTCN, tmp_path: object) -> None:
    path = str(tmp_path) + "/model.pt"  # type: ignore[operator]
    model.export_torchscript(path)
    loaded = torch.jit.load(path)
    x = torch.zeros(1, 12, 7)
    out = loaded(x)
    assert out.shape == (1, 2)
