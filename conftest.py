import os

# Set before any test module imports sentence_transformers / transformers.
# TensorFlow installed on this machine is compiled against NumPy 1.x and crashes
# under NumPy 2.x. Disabling TF backend keeps sentence-transformers PyTorch-only.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
