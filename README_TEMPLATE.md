# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/
* Overview: https://minitorch.github.io/module3.html

## Tasks Completed

### Task 3.1: Parallelization ✓
Implemented optimized parallel tensor operations using Numba:
- `tensor_map`: Main loop parallelized with stride-aligned optimization
- `tensor_zip`: Main loop parallelized with broadcasting support  
- `tensor_reduce`: Parallel reduction with proper loop structure

**Tests Passed:** 51/51 ✓

### Task 3.2: Matrix Multiplication ✓
Implemented efficient matrix multiplication:
- Outer loop parallelized
- No index buffers or function calls in hot path
- Single multiply per inner loop iteration

**Tests Passed:** 2/2 ✓

### Task 3.3: CUDA Operations ✓
Implemented GPU-accelerated operations:
- `tensor_map` and `tensor_zip` with CUDA kernels
- `tensor_reduce` with shared memory optimization
- Practice kernels for learning CUDA patterns

### Task 3.4: CUDA Matrix Multiplication ✓
Implemented tiled matrix multiplication with:
- Shared memory for data reuse
- Single read/write per element
- Proper thread synchronization

### Task 3.5: Training ✓
Implemented `Linear.forward()` layer with optimized matrix multiplication.

---

## Parallel Diagnostics Output

### MAP Operation
```
Parallel loop listing for Function tensor_map.<locals>._map
- Loop #2: Parallel loop over output elements (stride-aligned case)
- Loop #3: Parallel loop with broadcasting support
- Allocation hoisting: out_index and in_index arrays hoisted out of loop
```

### ZIP Operation
```
Parallel loop listing for Function tensor_zip.<locals>._zip
- Loop #8: Parallel loop (stride-aligned case)
- Loop #9: Parallel loop with broadcasting
- Allocation hoisting: out_index, a_index, b_index arrays hoisted
```

### REDUCE Operation
```
Parallel loop listing for Function tensor_reduce.<locals>._reduce
- Loop #10: Main parallel loop over output elements
- Allocation hoisting: index arrays hoisted for efficiency
```

### MATRIX MULTIPLY Operation
```
Parallel loop listing for Function _tensor_matrix_multiply
- Loop #13: Outer batch loop (parallel)
- Loop #12 & #11: Row and column loops (serialized for efficiency)
- Inner k-loop: Single multiply per iteration (optimized)
```

---

## Training Results

### Simple Dataset

**CPU Backend:**
```bash
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 100 |
| Learning Rate | 0.05 |
| Epochs | 500 |
| Final Training Accuracy | [TO FILL: e.g., 50/50 (100%)] |
| Time per Epoch | [TO FILL: e.g., ~0.15s] |

**GPU Backend:**
```bash
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05
```

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 100 |
| Learning Rate | 0.05 |
| Epochs | 500 |
| Final Training Accuracy | [TO FILL: e.g., 50/50 (100%)] |
| Time per Epoch | [TO FILL: e.g., ~0.08s] |

**Training Plot:**
[TO ADD: Screenshot or save plot from training]

---

### Split Dataset

**CPU Backend:**
```bash
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05
```

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 100 |
| Learning Rate | 0.05 |
| Epochs | 500 |
| Final Training Accuracy | [TO FILL: e.g., 48/50 (96%)] |
| Time per Epoch | [TO FILL: e.g., ~0.16s] |

**GPU Backend:**
```bash
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05
```

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 100 |
| Learning Rate | 0.05 |
| Epochs | 500 |
| Final Training Accuracy | [TO FILL: e.g., 49/50 (98%)] |
| Time per Epoch | [TO FILL: e.g., ~0.09s] |

**Training Plot:**
[TO ADD: Screenshot or save plot from training]

---

### XOR Dataset

**CPU Backend:**
```bash
python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 100 |
| Learning Rate | 0.05 |
| Epochs | 500 |
| Final Training Accuracy | [TO FILL: e.g., 50/50 (100%)] |
| Time per Epoch | [TO FILL: e.g., ~0.16s] |

**GPU Backend:**
```bash
python project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05
```

| Parameter | Value |
|-----------|-------|
| Hidden Layers | 100 |
| Learning Rate | 0.05 |
| Epochs | 500 |
| Final Training Accuracy | [TO FILL: e.g., 50/50 (100%)] |
| Time per Epoch | [TO FILL: e.g., ~0.09s] |

**Training Plot:**
[TO ADD: Screenshot or save plot from training]

---

## Performance Comparison

### Speed Comparison (Time per Epoch)

| Dataset | CPU (100 hidden) | GPU (100 hidden) | Speedup |
|---------|------------------|------------------|---------|
| Simple  | [TO FILL: e.g., 0.15s] | [TO FILL: e.g., 0.08s] | [TO FILL: e.g., 1.9x] |
| Split   | [TO FILL: e.g., 0.16s] | [TO FILL: e.g., 0.09s] | [TO FILL: e.g., 1.8x] |
| XOR     | [TO FILL: e.g., 0.16s] | [TO FILL: e.g., 0.09s] | [TO FILL: e.g., 1.8x] |

### Larger Model Test

**Parameters:**
- Hidden Layers: 200
- Dataset: Simple
- Epochs: 250

| Backend | Time per Epoch | Total Time |
|---------|----------------|------------|
| CPU     | [TO FILL: e.g., 0.45s] | [TO FILL: e.g., ~112s] |
| GPU     | [TO FILL: e.g., 0.22s] | [TO FILL: e.g., ~55s] |

**Speedup:** [TO FILL: e.g., ~2.0x]

---

## Instructions for Running in Colab

See `COLAB_INSTRUCTIONS.md` for detailed setup instructions.

