# Sprint Project Utilities

This directory contains utility functions and classes for managing memory and monitoring GPU usage in PyTorch projects.

## utils.py

The `utils.py` file provides several useful functions and classes for memory management and monitoring:

### Memory Management Functions

1. **print_memory_usage()**
   - Prints detailed memory usage information for all variables in the current namespace
   - Shows memory usage by variable type and total memory usage
   - Special handling for PyTorch tensors and datasets

2. **clear_memory(variables_to_keep=None, clear_tensors=True, clear_cache=True)**
   - Clears variables and releases GPU memory
   - Allows specifying which variables to keep
   - Can optionally clear PyTorch tensors and cache

3. **clear_tensor_memory(tensor_names=None, keep_model=True)**
   - Specifically clears PyTorch tensors to free up GPU memory
   - Can target specific tensors or clear all tensors
   - Option to preserve model tensors

4. **get_tensor_memory_usage()**
   - Returns detailed information about GPU tensor memory usage
   - Includes size, data type, device, and memory usage for each tensor
   - Results are sorted by memory usage

5. **calculate_tensor_memory(shape, dtype=t.float32)**
   - Calculates memory requirements for a tensor of specified shape and type
   - Returns memory usage in bytes, KB, MB, and GB
   - Useful for planning memory allocation

### Memory Monitoring Class

**MemoryMonitor**
A class for monitoring GPU memory usage over time:

- `start()`: Begins memory monitoring
- `measure(label=None)`: Takes a memory measurement with optional label
- `plot()`: Creates a visualization of memory usage over time
- `reset()`: Resets the monitor's measurements
- `start_continuous_monitoring(interval=10, print_msg=False)`: Starts continuous monitoring
- `stop_continuous_monitoring()`: Stops the continuous monitoring

## Usage Example

```python
from utils import MemoryMonitor

# Initialize memory monitor
monitor = MemoryMonitor("Training")

# Start monitoring
monitor.start()

# Take measurements during training
monitor.measure("Before forward pass")
# ... training code ...
monitor.measure("After forward pass")

# Plot results
monitor.plot()
```

## Requirements

- PyTorch
- Matplotlib
- CUDA-capable GPU (for GPU memory monitoring) 