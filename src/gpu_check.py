try:
    import tensorflow
except ImportError:
    print("Tensorflow is not installed, install requied packages from requirements.txt")
    exit(1)

import tensorflow

print("If you see the tensor result, then the Tensorflow is available.")
rs = tensorflow.reduce_sum(tensorflow.random.normal([1000, 1000]))
print(rs)

gpus = tensorflow.config.list_physical_devices('GPU')
if len(gpus) == 0:
    print("No GPU available.")
else:
    print(f"GPUs available: {len(gpus)}")
    print(gpus)
