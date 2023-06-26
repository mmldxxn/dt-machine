import cupy as cp

# Create input arrays on the GPU
x_gpu = cp.array([1, 2, 3])
y_gpu = cp.array([4, 5, 6])

# Perform vector addition on the GPU
result_gpu = x_gpu + y_gpu

# Transfer the result back to the CPU
result_cpu = cp.asnumpy(result_gpu)

print("Computed on GPU:", result_cpu)

