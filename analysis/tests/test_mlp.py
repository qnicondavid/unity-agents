import torch
import pytest
from analysis.mlp.network import (
    fit_scaler,
    apply_scaler,
    MLP,
    mean_squared_error,
    train_network,
    predict,
)


def test_scaler_fit():
    X = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float32)
    means, sds = fit_scaler(X)
    
    assert means.shape == (2,)
    assert sds.shape == (2,)
    assert torch.allclose(means, torch.tensor([2.0, 20.0]))


def test_scaler_normalization():
    X = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float32)
    means, sds = fit_scaler(X)
    X_scaled = apply_scaler(X, means, sds)
    
    assert torch.allclose(X_scaled.mean(dim=0), torch.zeros(2), atol=1e-5)


def test_mlp_forward_pass():
    mlp = MLP(n_inputs=5, hidden_sizes=[32], n_outputs=1)
    X = torch.randn(10, 5)
    output = mlp(X)
    
    assert output.shape == (10, 1)


def test_mlp_different_architectures():
    for hidden_sizes in [[], [16], [64, 32]]:
        mlp = MLP(n_inputs=5, hidden_sizes=hidden_sizes, n_outputs=1)
        X = torch.randn(10, 5)
        output = mlp(X)
        
        assert output.shape == (10, 1)


def test_mse_loss_zero():
    outputs = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([[1.0], [2.0], [3.0]])
    loss = mean_squared_error(outputs, targets)
    
    assert torch.allclose(loss, torch.tensor(0.0))


def test_mse_loss_positive():
    outputs = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([[2.0], [3.0], [4.0]])
    loss = mean_squared_error(outputs, targets)
    
    assert torch.allclose(loss, torch.tensor(1.0))


def test_train_network_basic():
    rows = [[float(i)] for i in range(20)]
    targets = [float(i * 2) for i in range(20)]
    
    model, x_means, x_sds, y_means, y_sds, (train_losses, val_losses) = train_network(
        rows, targets,
        hidden_sizes=[32],
        lr=0.1,
        batch_size=4,
        epochs=5,
        seed=1
    )
    
    assert isinstance(model, MLP)
    assert len(train_losses) == 5


def test_predict_shape():
    rows = [[float(i)] for i in range(20)]
    targets = [float(i * 2) for i in range(20)]
    
    trained = train_network(rows, targets, hidden_sizes=[32],
                           lr=0.1, batch_size=4, epochs=3, seed=1)[:5]
    
    predictions = predict(trained, [[5.0], [10.0]])
    assert len(predictions) == 2


def test_mlp_on_linear_data():
    rows = [[float(i)] for i in range(30)]
    targets = [float(i * 2 + 1) for i in range(30)]
    
    trained = train_network(rows, targets, hidden_sizes=[32],
                           lr=0.1, batch_size=4, epochs=15, seed=1)[:5]
    
    pred_low = predict(trained, [[0.0]])[0]
    pred_high = predict(trained, [[29.0]])[0]
    
    assert pred_low < pred_high


def test_mlp_validation_loss():
    rows = [[float(i)] for i in range(20)]
    targets = [float(i * 2) for i in range(20)]
    val_rows = [[float(i + 0.5)] for i in range(10)]
    val_targets = [float((i + 0.5) * 2) for i in range(10)]
    
    model, x_means, x_sds, y_means, y_sds, (train_losses, val_losses) = train_network(
        rows, targets,
        hidden_sizes=[32],
        lr=0.1,
        batch_size=4,
        epochs=5,
        seed=1,
        validation_rows=val_rows,
        validation_targets=val_targets
    )
    
    assert len(val_losses) == 5
