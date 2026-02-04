# Setup Instructions

Complete setup guide for running the Grokking Detection project from a clean machine.

## Prerequisites

- Python 3.8 or higher
- Git (optional, for version control)
- No GPU required (CPU-only PyTorch)

## Step-by-Step Setup

### 1. Navigate to Project Directory

```bash
cd D:\Nishal\Grokking
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

This creates a new virtual environment in the `venv/` directory.

### 3. Activate Virtual Environment

**Windows (Command Prompt):**
```bash
venv\Scripts\activate
```

**Windows (PowerShell):**
```bash
venv\Scripts\Activate.ps1
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

You should see `(venv)` prefix in your terminal prompt.

### 4. Upgrade pip (Optional but Recommended)

```bash
python -m pip install --upgrade pip
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including:
- **PyTorch 2.10.0+cpu** (CPU-only, no CUDA)
- NumPy 2.4.1
- Matplotlib 3.10.8
- Seaborn 0.13.2
- SciPy 1.17.0
- tqdm 4.67.1
- All other dependencies

**Installation time:** ~2-5 minutes depending on your internet connection.

### 6. Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

Expected output:
```
PyTorch 2.10.0+cpu installed successfully
CUDA available: False
```

### 7. Run Test Script (Optional)

Test that the model can be initialized:

```bash
python -c "from config import ModelConfig; from src.model import DecoderOnlyTransformer; model = DecoderOnlyTransformer(ModelConfig()); print(f'Model initialized with {model.count_parameters():,} parameters')"
```

Expected output:
```
Model initialized with 3,185,920 parameters
```

## Running the Experiment

### Start Training

```bash
python scripts/train.py
```

**Note:** Training on CPU will be significantly slower than GPU. For the full 200,000 steps:
- **GPU (e.g., RTX 3090):** ~3 hours
- **CPU (modern multi-core):** ~24-48 hours

For testing, you can reduce training steps in `config.py`:
```python
num_steps: int = 10_000  # Instead of 200_000
```

### Generate Visualizations

After training completes:

```bash
python scripts/visualize.py
```

## Deactivating Virtual Environment

When you're done working:

```bash
deactivate
```

## Starting a New Session

Every time you open a new terminal to work on this project:

1. Navigate to project directory: `cd D:\Nishal\Grokking`
2. Activate virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Linux/macOS)
3. Run your commands
4. Deactivate when done: `deactivate`

## Troubleshooting

### Issue: "python: command not found"
**Solution:** Use `python3` instead of `python`:
```bash
python3 -m venv venv
```

### Issue: "Cannot activate virtual environment on PowerShell"
**Solution:** Enable script execution:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Solution:** Make sure the virtual environment is activated (you should see `(venv)` in your prompt). If still failing, reinstall dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "pip install fails with SSL certificate error"
**Solution:** Try with trusted host:
```bash
pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Issue: Training is too slow on CPU
**Solution:**
1. Reduce batch size in `config.py`: `batch_size: int = 64` (instead of 256)
2. Reduce training steps: `num_steps: int = 50_000` (instead of 200,000)
3. Reduce model size: `d_model: int = 128`, `d_ff: int = 512`

## Clean Installation on a New Machine

From scratch on a machine without the project:

```bash
# 1. Clone/copy project to local directory
cd D:\Nishal\Grokking

# 2. Create virtual environment
python -m venv venv

# 3. Activate it
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Linux/macOS

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run training
python scripts/train.py
```

## Package Versions

This project uses the following key package versions (from `requirements.txt`):

| Package | Version | Purpose |
|---------|---------|---------|
| torch | 2.10.0+cpu | Deep learning framework (CPU-only) |
| numpy | 2.4.1 | Numerical computations |
| matplotlib | 3.10.8 | Plotting and visualization |
| seaborn | 0.13.2 | Statistical visualization |
| scipy | 1.17.0 | Scientific computing (SVD) |
| tqdm | 4.67.1 | Progress bars |

## Notes

- **No CUDA dependencies:** This installation is CPU-only and will work on any machine without a GPU.
- **Virtual environment is isolated:** Packages installed in `venv/` do not affect your system Python installation.
- **Reproducibility:** The `requirements.txt` file pins exact versions to ensure consistent behavior across machines.
- **Storage:** The virtual environment takes ~1-2 GB of disk space.

## Uninstalling

To completely remove the project:

1. Deactivate virtual environment: `deactivate`
2. Delete the entire project directory (includes `venv/`)
3. Done! No system-wide changes were made.
