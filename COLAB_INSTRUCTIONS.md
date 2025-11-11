# Google Colab Setup and Training Instructions

## Step 1: Setup Colab Environment

### 1.1 Upload Your Repository
```python
# Option A: Clone from GitHub (if you've pushed your code)
!git clone https://github.com/YOUR_USERNAME/minitorch-module-3-fedickinson.git
%cd minitorch-module-3-fedickinson

# Option B: Upload ZIP file
# 1. Zip your local repository
# 2. Upload to Colab using the files panel
# 3. Unzip:
!unzip minitorch-module-3-fedickinson.zip
%cd minitorch-module-3-fedickinson
```

### 1.2 Install Dependencies
```python
# numba-cuda is already pre-installed in Colab (check with: pip list | grep numba)
# Just install other requirements
!pip install -r requirements.txt

# Install minitorch package
!pip install -e .
```

**Note:** Google Colab comes with `numba-cuda`, `numba`, and `numpy` pre-installed with compatible versions. The `requirements.txt` file only specifies testing tools and other dependencies.

### 1.3 Enable GPU
1. Go to Runtime → Change runtime type
2. Select "T4 GPU" or "A100 GPU"
3. Click Save

### 1.4 Verify CUDA
```python
import numba.cuda
print("CUDA Available:", numba.cuda.is_available())
```

---

## Step 2: Run Training Experiments

### 2.1 Simple Dataset - CPU
```python
!python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05 --PTS 50
```

**What to record:**
- Last epoch line showing: `Epoch XXX | Loss: X.XXXX | Correct: XX/50 | Time/Epoch: X.XXXXs`
- Note the Time/Epoch value (last 10 epochs average)

### 2.2 Simple Dataset - GPU
```python
!python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET simple --RATE 0.05 --PTS 50
```

**What to record:**
- Final accuracy (Correct: XX/50)
- Time per epoch
- Calculate speedup: CPU_time / GPU_time

---

### 2.3 Split Dataset - CPU
```python
!python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split --RATE 0.05 --PTS 50
```

### 2.4 Split Dataset - GPU
```python
!python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET split --RATE 0.05 --PTS 50
```

---

### 2.5 XOR Dataset - CPU
```python
!python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET xor --RATE 0.05 --PTS 50
```

### 2.6 XOR Dataset - GPU
```python
!python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 100 --DATASET xor --RATE 0.05 --PTS 50
```

---

## Step 3: Larger Model Test

### 3.1 Bigger Hidden Layers - CPU
```python
!python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 200 --DATASET simple --RATE 0.05 --PTS 50
```

### 3.2 Bigger Hidden Layers - GPU
```python
!python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 200 --DATASET simple --RATE 0.05 --PTS 50
```

---

## Step 4: Record Results

### Template for Each Run

Create a table like this for each experiment:

```
Dataset: [simple/split/xor]
Backend: [CPU/GPU]
Hidden Layers: [100/200]
Learning Rate: 0.05
Points: 50
Epochs: 500

Results:
- Final Accuracy: [XX/50]
- Avg Time per Epoch: [X.XXXs]
- Total Training Time: [~XXXs]

Final Output Line:
[Copy the last line from training, e.g.:]
Epoch 499 | Loss:   2.6950 | Correct:  50/50 | Time/Epoch: 0.0517s
```

---

## Step 5: Tips for Colab

### Save Long Outputs
```python
# Redirect output to file
!python3 project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET simple --RATE 0.05 > simple_cpu_output.txt 2>&1

# View last 20 lines
!tail -20 simple_cpu_output.txt
```

### Run Multiple Experiments in Sequence
```python
datasets = ['simple', 'split', 'xor']
backends = ['cpu', 'gpu']

for dataset in datasets:
    for backend in backends:
        print(f"\n{'='*60}")
        print(f"Running {dataset.upper()} dataset on {backend.upper()}")
        print(f"{'='*60}\n")
        
        !python3 project/run_fast_tensor.py \
            --BACKEND {backend} \
            --HIDDEN 100 \
            --DATASET {dataset} \
            --RATE 0.05 \
            --PTS 50 \
            2>&1 | tail -15
        
        print("\n")
```

### Monitor GPU Usage
```python
# In a separate cell, run periodically:
!nvidia-smi
```

---

## Step 6: Expected Performance Targets

Based on the assignment requirements:

| Metric | Target |
|--------|--------|
| CPU Time/Epoch | < 2 seconds |
| GPU Time/Epoch | < 1 second |
| GPU Speedup | ~2-10x vs CPU |

**Note:** Actual performance depends on:
- Colab GPU type (T4 vs A100)
- Current Colab load
- Dataset complexity
- Model size

---

## Step 7: Troubleshooting

### If you get "CUDA_ERROR_UNSUPPORTED_PTX_VERSION":
This is a PTX version mismatch. **Solution:**
```python
# Verify numba-cuda is installed (should be pre-installed in Colab)
!pip list | grep -E "numba|numpy"

# If missing, install it:
!pip install numba-cuda

# Then reinstall your package
!pip install -e .
```

### If CUDA is not available:
```python
# Check CUDA installation
!nvcc --version

# Verify GPU is enabled
# Go to Runtime → Change runtime type → Select GPU

# Check CUDA availability
from numba import cuda
print("CUDA Available:", cuda.is_available())
```

### If training crashes:
```python
# Try smaller batch sizes or hidden layers
!python3 project/run_fast_tensor.py --BACKEND gpu --HIDDEN 50 --DATASET simple --RATE 0.05 --PTS 25
```

### If GPU runs slower than CPU:
- This can happen with very small models
- GPU overhead dominates for small computations
- Use larger models (hidden=200+) to see GPU benefits

---

## Step 8: Collecting Results for README

After running all experiments, create a summary table:

```python
# Example results structure
results = {
    'simple': {
        'cpu': {'time': 0.15, 'accuracy': '50/50'},
        'gpu': {'time': 0.08, 'accuracy': '50/50'}
    },
    'split': {
        'cpu': {'time': 0.16, 'accuracy': '48/50'},
        'gpu': {'time': 0.09, 'accuracy': '49/50'}
    },
    'xor': {
        'cpu': {'time': 0.16, 'accuracy': '50/50'},
        'gpu': {'time': 0.09, 'accuracy': '50/50'}
    }
}

# Calculate speedups
for dataset in results:
    cpu_time = results[dataset]['cpu']['time']
    gpu_time = results[dataset]['gpu']['time']
    speedup = cpu_time / gpu_time
    print(f"{dataset}: {speedup:.2f}x speedup")
```

Fill in the `README_TEMPLATE.md` with your actual results!

