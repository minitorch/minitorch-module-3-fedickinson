import random

import numba

import minitorch

datasets = minitorch.datasets
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = None
if numba.cuda.is_available():
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def RParam(*shape, backend):
    r = 2 * (minitorch.rand(shape, backend=backend) - 0.5)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden, backend):
        super().__init__()

        # Submodules
        self.layer1 = Linear(2, hidden, backend)
        self.layer2 = Linear(hidden, hidden, backend)
        self.layer3 = Linear(hidden, 1, backend)

    def forward(self, x):
        # 3 layer network with relu
        h1 = self.layer1.forward(x).relu()
        h2 = self.layer2.forward(h1).relu()
        return self.layer3.forward(h2).sigmoid()


class Linear(minitorch.Module):
    def __init__(self, in_size, out_size, backend):
        super().__init__()
        self.weights = RParam(in_size, out_size, backend=backend)
        self.bias = RParam(out_size, backend=backend)
        self.out_size = out_size

    def forward(self, x):
        # matrix multiply + bias
        return x @ self.weights + self.bias


class FastTrain:
    def __init__(self, hidden_layers, backend=FastTensorBackend):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers, backend)
        self.backend = backend

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.model = Network(self.hidden_layers, self.backend)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []

        X = minitorch.tensor(data.X, backend=self.backend)
        y = minitorch.tensor(data.y, backend=self.backend)

        for epoch in range(max_epochs):
            total_loss = 0.0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--PTS", type=int, default=50, help="number of points")
    parser.add_argument("--HIDDEN", type=int, default=10, help="number of hiddens")
    parser.add_argument("--RATE", type=float, default=0.05, help="learning rate")
    parser.add_argument("--BACKEND", default="cpu", help="backend mode")
    parser.add_argument("--DATASET", default="simple", help="dataset")
    parser.add_argument("--PLOT", default=False, help="dataset")

    args = parser.parse_args()

    PTS = args.PTS

    if args.DATASET == "xor":
        data = minitorch.datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        data = minitorch.datasets["Simple"](PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](PTS)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    # Select backend
    if args.BACKEND == "gpu":
        if GPUBackend is None:
            print("GPU backend requested but CUDA is not available. Falling back to CPU.")
            backend = FastTensorBackend
        else:
            backend = GPUBackend
    else:
        backend = FastTensorBackend

    FastTrain(HIDDEN, backend=backend).train(data, RATE)
