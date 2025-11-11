import random
import sys
import os

# Configure Numba threading BEFORE importing numba
# Use single thread to avoid deadlocks in Kaggle/Colab
os.environ['NUMBA_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['NUMBA_THREADING_LAYER'] = 'workqueue'

import numba

import minitorch

datasets = minitorch.datasets

# Debug: Print before creating backends
print("Creating FastTensorBackend...", flush=True)
FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
print("✅ FastTensorBackend created", flush=True)

GPUBackend = None
if numba.cuda.is_available():
    print("Creating GPUBackend...", flush=True)
    GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)
    print("✅ GPUBackend created", flush=True)


def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


def RParam(*shape, backend):
    print(f"  RParam: creating random tensor with shape {shape}...", flush=True)
    r = 2 * (minitorch.rand(shape, backend=backend) - 0.5)
    print(f"  ✅ RParam: tensor created", flush=True)
    return minitorch.Parameter(r)


class Network(minitorch.Module):
    def __init__(self, hidden, backend):
        print(f"Network.__init__: hidden={hidden}", flush=True)
        super().__init__()

        # Submodules
        print(f"Creating layer1: Linear(2, {hidden})...", flush=True)
        self.layer1 = Linear(2, hidden, backend)
        print(f"✅ layer1 created", flush=True)
        
        print(f"Creating layer2: Linear({hidden}, {hidden})...", flush=True)
        self.layer2 = Linear(hidden, hidden, backend)
        print(f"✅ layer2 created", flush=True)
        
        print(f"Creating layer3: Linear({hidden}, 1)...", flush=True)
        self.layer3 = Linear(hidden, 1, backend)
        print(f"✅ layer3 created", flush=True)

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
        # x shape: (batch, in_size)
        # weights shape: (in_size, out_size)
        # result shape: (batch, out_size)
        batch = x.shape[0]
        return (x @ self.weights.value).view(batch, self.out_size) + self.bias.value.view(1, self.out_size)


class FastTrain:
    def __init__(self, hidden_layers, backend=FastTensorBackend):
        print(f"FastTrain.__init__: hidden_layers={hidden_layers}", flush=True)
        self.hidden_layers = hidden_layers
        print(f"Creating Network model...", flush=True)
        self.model = Network(hidden_layers, backend)
        print(f"✅ Network model created", flush=True)
        self.backend = backend

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x], backend=self.backend))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X, backend=self.backend))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        import time
        print(f"Creating model with {self.hidden_layers} hidden layers...", flush=True)
        self.model = Network(self.hidden_layers, self.backend)
        print("✅ Model created", flush=True)
        
        optim = minitorch.SGD(self.model.parameters(), learning_rate)
        losses = []

        print(f"Creating input tensors (N={data.N})...", flush=True)
        X = minitorch.tensor(data.X, backend=self.backend)
        y = minitorch.tensor(data.y, backend=self.backend)
        print("✅ Tensors created", flush=True)

        print(f"Starting training for {max_epochs} epochs...", flush=True)
        epoch_times = []
        for epoch in range(max_epochs):
            if epoch == 0:
                print("Starting epoch 0 (first epoch may be slow due to JIT compilation)...", flush=True)
            
            start_time = time.time()
            
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
            
            epoch_time = time.time() - start_time
            epoch_times.append(epoch_time)

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs - 1:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                avg_time = sum(epoch_times[-10:]) / len(epoch_times[-10:])
                print(f"Epoch {epoch:3d} | Loss: {total_loss:8.4f} | Correct: {correct:3d}/{data.N} | Time/Epoch: {avg_time:.4f}s", flush=True)


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
    
    print(f"Loading dataset: {args.DATASET} with {PTS} points...", flush=True)
    if args.DATASET == "xor":
        data = minitorch.datasets["Xor"](PTS)
    elif args.DATASET == "simple":
        data = minitorch.datasets["Simple"](PTS)
    elif args.DATASET == "split":
        data = minitorch.datasets["Split"](PTS)
    elif args.DATASET == "circle":
        data = minitorch.datasets["Circle"](PTS)
    elif args.DATASET == "spiral":
        data = minitorch.datasets["Spiral"](PTS)
    print(f"✅ Dataset loaded: {data.N} points", flush=True)

    HIDDEN = int(args.HIDDEN)
    RATE = args.RATE

    # Select backend
    print(f"Selecting backend: {args.BACKEND}", flush=True)
    if args.BACKEND == "gpu":
        if GPUBackend is None:
            print("⚠️ GPU backend requested but CUDA is not available. Falling back to CPU.", flush=True)
            backend = FastTensorBackend
        else:
            print("✅ Using GPU backend", flush=True)
            backend = GPUBackend
    else:
        print("✅ Using CPU FastOps backend", flush=True)
        backend = FastTensorBackend

    print(f"\n{'='*60}", flush=True)
    print(f"Configuration: HIDDEN={HIDDEN}, RATE={RATE}, BACKEND={args.BACKEND}", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    FastTrain(HIDDEN, backend=backend).train(data, RATE)
