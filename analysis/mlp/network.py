# The comments in the code are based on my (ND Milandru) iai lecture notes.

import torch


def fit_scaler(X):
    means = X.mean(dim=0)
    sds = X.std(dim=0, correction=0)
    sds = torch.where(sds == 0, torch.ones_like(sds), sds)
    return (means, sds)


def apply_scaler(X, means, sds):
    return (X - means) / sds


class RunDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (self.X[i], self.y[i])


# - mini-batch: non-overlapping subsets of the training data  [Lecture 05, slide 32]
def make_loader(X, y, batch_size, seed):
    dataset = RunDataset(X, y)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        drop_last=len(dataset) % batch_size == 1
    )
    return loader


# Artificial neuron  [Lecture 05, slides 7-8]
#     o_j = phi(sum_{i=1..n} w_ij x_i - theta_j)
# - Weights: how much each input counts (can be + or -)
# - Transfer function: adds up the weighted inputs
# - Activation function: turns the sum into the output
# GPUs  [Lecture 05, slide 35]
# - neural network layers are essentially large matrices
#     A = phi(W X + b)
# where W = weight matrix (one row per neuron), X = inputs (one column per example),
#   b = biases (one per neuron), phi = activation function, A = the layer's outputs
#   (one column per example)
class MLP(torch.nn.Module):
    # ReLU(z) = max(0, z)  [Lecture 05 notes, practice problem]
    # - dropout: a percentage of randomly selected neurons are bypassed at each epoch
    #   (happens during training only)  [Lecture 05, slide 29]
    def __init__(self, n_inputs, hidden_sizes, n_outputs, activation=torch.nn.ReLU, dropout=0.0):
        super().__init__()

        self.layers = torch.nn.ModuleList()

        in_features = n_inputs

        for hidden_size in hidden_sizes:
            self.layers.append(torch.nn.Linear(in_features, hidden_size))
            self.layers.append(torch.nn.BatchNorm1d(hidden_size))
            self.layers.append(activation())

            if dropout > 0:
                self.layers.append(torch.nn.Dropout(dropout))

            in_features = hidden_size

        self.layers.append(torch.nn.Linear(in_features, n_outputs))

    # Multi-layered perceptron (MLP)  [Lecture 05, slide 9]
    #     y = f3(f2(f1(x)))
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


# Loss / error / cost function: degree of incorrectness  [Lecture 05, slides 11-14]
# - how far* is the result of the network from the target
def mean_squared_error(outputs, targets):
    return ((outputs - targets) ** 2).mean()


# Backpropagation  [Lecture 05, slides 16-18]
# - with a layered function like y, the chain rule can be applied to each function separately
# - rule: move each weight against the slope (slope > 0 -> decrease w, slope < 0 -> increase w)
# Gradient descent  [Lecture 05, slides 19-20]
# - goal: repeat backprop (compute the slope of every weight) + update (step each weight against
#   its slope) until the slope ~ 0, i.e. the loss stops going down
# - works the same for one weight or for many: every weight is updated in the same step,
#   each with its own slope
#     w_new = w_old - eta * dL/dw
# where eta = learning rate (the step size, chosen by you)
def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    total_examples = 0
    for Xb, yb in loader:
        outputs = model(Xb)
        loss = mean_squared_error(outputs, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.shape[0]
        total_examples += Xb.shape[0]
    return total_loss / total_examples


def evaluate(model, X, batch_size=1024):
    ans = []
    model.eval()
    with torch.no_grad():
        for start in range(0, X.shape[0], batch_size):
            Xb = X[start:start + batch_size]
            ans.append(model(Xb))
    return torch.cat(ans, dim=0)


# Training a neural network  [Lecture 05, slide 23]
# - monitor the error function (loss)
# - if everything goes well, it should decrease
# - epoch: one pass through the entire dataset, exactly once
# Potential outcomes  [Lecture 05, slide 24]
# - convergence: the model fits the data, validation loss is similar to training loss and both
#   are low
# - overfitting: the model overfits the data: validation loss is higher than training loss
# - underfitting: the model underfits the data: validation loss and training loss are both high
def fit(model, loader, optimizer, epochs, X_val=None, y_val=None):
    train_losses = []
    validation_losses = []
    for _ in range(epochs):
        train_loss = train_epoch(model, loader, optimizer)
        train_losses.append(train_loss)
        if X_val is not None:
            validation_outputs = evaluate(model, X_val)
            validation_loss = mean_squared_error(validation_outputs, y_val)
            validation_losses.append(validation_loss.item())
    return (train_losses, validation_losses)


# Main elements in a neural network  [Lecture 05, slide 21]
# - layers
# - connections
# - input
# - output
# - optimizer
# - regularizers
def train_network(rows, targets, hidden_sizes, lr, batch_size, epochs, seed,
                  activation=torch.nn.ReLU, dropout=0.0, validation_rows=None,
                  validation_targets=None):
    torch.set_num_threads(1)
    X = torch.tensor(rows, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).reshape(-1, 1)
    x_means, x_sds = fit_scaler(X)
    y_means, y_sds = fit_scaler(y)
    X_scaled = apply_scaler(X, x_means, x_sds)
    y_scaled = apply_scaler(y, y_means, y_sds)
    X_val = None
    y_val = None
    if validation_rows is not None:
        X_val = apply_scaler(torch.tensor(validation_rows, dtype=torch.float32), x_means, x_sds)
        y_val = torch.tensor(validation_targets, dtype=torch.float32).reshape(-1, 1)
        y_val = apply_scaler(y_val, y_means, y_sds)
    loader = make_loader(X_scaled, y_scaled, batch_size, seed)
    torch.manual_seed(seed)
    model = MLP(X.shape[1], hidden_sizes, 1, activation, dropout)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    losses = fit(model, loader, optimizer, epochs, X_val, y_val)
    return (model, x_means, x_sds, y_means, y_sds, losses)


def predict(trained, rows):
    model = trained[0]
    x_means = trained[1]
    x_sds = trained[2]
    y_means = trained[3]
    y_sds = trained[4]

    X = torch.tensor(rows, dtype=torch.float32)
    X_scaled = apply_scaler(X, x_means, x_sds)

    outputs = evaluate(model, X_scaled)
    return (outputs * y_sds + y_means).reshape(-1).tolist()
