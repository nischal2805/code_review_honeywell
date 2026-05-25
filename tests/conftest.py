import os

# Prevent transformers from loading TensorFlow (broken against NumPy 2.x on this system).
# sentence-transformers only needs PyTorch; TF import causes AttributeError/_ARRAY_API errors.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
