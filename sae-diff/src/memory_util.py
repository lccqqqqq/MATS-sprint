import torch as t
import gc
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import inspect
from nnsight import LanguageModel
from rich import print as rprint
from rich.table import Table
import os




def print_memory_usage():
    # Dictionary to store memory by type
    memory_by_type = defaultdict(int)

    # Create a copy of the locals dictionary to avoid the "dictionary changed size during iteration" error
    locals_copy = dict(locals())

    # Iterate through all variables in the current namespace
    for name, obj in locals_copy.items():
        # Skip internal/system variables
        if name.startswith('_'):
            continue
            
        # Get size in MB
        size_mb = sys.getsizeof(obj) / (1024 * 1024)
        
        # For PyTorch tensors, get actual memory allocation
        if isinstance(obj, t.Tensor):
            size_mb = obj.element_size() * obj.nelement() / (1024 * 1024)
            
        # For datasets, estimate size
        if str(type(obj).__module__).startswith('datasets'):
            size_mb = sum(sys.getsizeof(x) for x in obj.values()) / (1024 * 1024)
        
        type_name = type(obj).__name__
        memory_by_type[type_name] += size_mb
        
        print(f"Variable: {name:<20} Type: {type_name:<20} Size: {size_mb:.2f} MB")

    print("\nTotal memory by type:")
    for type_name, total_mb in memory_by_type.items():
        print(f"{type_name:<20}: {total_mb:.2f} MB")


def clear_memory(variables_to_keep=None, clear_tensors=True, clear_cache=True):
    """
    Clears variables and releases GPU memory.
    
    Args:
        variables_to_keep: List of variable names to keep (default: None, keeps all)
        clear_tensors: Whether to clear PyTorch tensors (default: True)
        clear_cache: Whether to clear PyTorch cache (default: True)
    
    Returns:
        None
    """
    import gc
    
    # If no specific variables to keep, use an empty list
    if variables_to_keep is None:
        variables_to_keep = []
    
    # Get all variables in the current namespace
    current_vars = dict(locals())
    
    # Clear PyTorch cache
    if clear_cache and 't' in current_vars:
        t.cuda.empty_cache()
    
    # Run garbage collection
    gc.collect()
    
    # Clear variables that are not in the keep list
    for var_name, var in current_vars.items():
        # Skip internal variables and variables to keep
        if var_name.startswith('_') or var_name in variables_to_keep:
            continue
        
        # Delete the variable
        if var_name in locals():
            del locals()[var_name]
    
    # Print memory usage after clearing
    print_gpu_memory("after clearing memory")
    
    return None

def clear_tensor_memory(tensor_names=None, keep_model=True):
    """
    Specifically clears PyTorch tensors to free up GPU memory.
    
    Args:
        tensor_names: List of tensor variable names to clear (default: None, clears all tensors)
        keep_model: Whether to keep the model tensor (default: True)
    
    Returns:
        None
    """
    # Get all variables in the current namespace
    current_vars = dict(locals())
    
    # If no specific tensors to clear, clear all tensors
    if tensor_names is None:
        tensor_names = []
        for var_name, var in current_vars.items():
            if isinstance(var, t.Tensor) and (var_name != 'model' or not keep_model):
                tensor_names.append(var_name)
    
    # Clear specified tensors
    for tensor_name in tensor_names:
        if tensor_name in locals():
            # Move tensor to CPU first to ensure proper cleanup
            if hasattr(locals()[tensor_name], 'cpu'):
                locals()[tensor_name] = locals()[tensor_name].cpu()
            # Delete the tensor
            del locals()[tensor_name]
    
    # Clear PyTorch cache
    t.cuda.empty_cache()
    
    # Run garbage collection
    import gc
    gc.collect()
    
    # Print memory usage after clearing
    print_gpu_memory("after clearing tensor memory")
    
    return None

# def get_tensor_memory_usage():
#     """
#     Returns a dictionary of tensor memory usage in the current namespace.
    
#     Returns:
#         dict: Dictionary mapping tensor names to their memory usage in MB
#     """
#     # Get all variables in the current namespace
#     current_vars = dict(locals())
    
#     # Dictionary to store tensor memory usage
#     tensor_memory = {}
    
#     # Calculate memory usage for each tensor
#     for var_name, var in current_vars.items():
#         if isinstance(var, t.Tensor):
#             # Calculate memory in MB
#             memory_mb = var.element_size() * var.nelement() / (1024 * 1024)
#             tensor_memory[var_name] = memory_mb
    
#     return tensor_memory

# # Example usage:
# # clear_memory(variables_to_keep=['model', 'ds_filtered'])
# # clear_tensor_memory(tensor_names=['logits', 'cache'])
# # tensor_memory = get_tensor_memory_usage()
# # print(tensor_memory)


def get_tensor_memory_usage():
    # Dictionary to store variable names and their memory usage
    memory_usage = {}
    
    # Get all variables in the current scope
    for var_name, var in globals().items():
        # Skip built-in variables and modules
        if var_name.startswith('__') or var_name in ('get_tensor_memory_usage', 'torch', 'np', 'sys'):
            continue
            
        # Check if it's a PyTorch tensor on GPU
        if isinstance(var, t.Tensor) and var.is_cuda:
            # Calculate memory in MB
            memory_mb = var.element_size() * var.nelement() / (1024 * 1024)
            memory_usage[var_name] = {
                'size': var.size(),
                'dtype': var.dtype,
                'device': var.device,
                'memory_mb': memory_mb
            }
    
    # Sort by memory usage (descending)
    sorted_usage = dict(sorted(memory_usage.items(), key=lambda x: x[1]['memory_mb'], reverse=True))
    return sorted_usage

# Example usage:
# # Print all GPU tensors and their memory usage
# gpu_memory_usage = get_tensor_memory_usage()
# for name, info in gpu_memory_usage.items():
#     print(f"{name}: {info['size']} ({info['dtype']}) on {info['device']} - {info['memory_mb']:.2f} MB")


# # Print total GPU memory usage
# total_memory = sum(info['memory_mb'] for info in gpu_memory_usage.values())
# print(f"\nTotal GPU memory used by Python variables: {total_memory:.2f} MB")



def calculate_tensor_memory(shape, dtype=t.float32):
    """
    Calculates the memory required for a tensor of a specific size and data type.
    
    Args:
        shape: Shape of the tensor (tuple or list of integers)
        dtype: Data type of the tensor (default: t.float32)
    
    Returns:
        dict: Dictionary with memory information in different units
    """
    # Calculate number of elements
    num_elements = 1
    for dim in shape:
        num_elements *= dim
    
    # Get element size in bytes
    element_size = t.tensor([], dtype=dtype).element_size()
    
    # Calculate memory in different units
    memory_bytes = num_elements * element_size
    memory_kb = memory_bytes / 1024
    memory_mb = memory_kb / 1024
    memory_gb = memory_mb / 1024
    
    return {
        'bytes': memory_bytes,
        'kb': memory_kb,
        'mb': memory_mb,
        'gb': memory_gb,
        'num_elements': num_elements,
        'element_size': element_size,
        'dtype': str(dtype)
    }

# Example usage:
# memory_info = calculate_tensor_memory((2000, 4096, 200), dtype=t.bfloat16)
# print(f"Memory required: {memory_info['gb']:.2f} GB")

class MemoryMonitor:
    """
    A class to monitor GPU memory usage over time.
    """
    def __init__(self, name="monitor"):
        """
        Initialize the memory monitor.
        
        Args:
            name: Name of the monitor (default: "Memory Monitor")
        """
        self.name = name
        self.measurements = []
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """Start monitoring memory usage."""
        import time
        self.start_time = time.time()
        self.start_memory = t.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        self.measurements.append((0, self.start_memory, 0, "Start"))
        print(f"{self.name}: Started monitoring at {self.start_memory:.2f} GB")
        
    def measure(self, label=None, print_msg=True):
        """
        Take a measurement of current memory usage.
        
        Args:
            label: Label for this measurement (default: None)
        """
        if self.start_time is None:
            print(f"{self.name}: Monitoring not started. Call start() first.")
            return
        
        import time
        current_time = time.time() - self.start_time
        current_memory = t.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
        memory_change = current_memory - self.measurements[-1][1]
        
        self.measurements.append((current_time, current_memory, memory_change, label if label is not None else ""))
        if print_msg:
            if label:
                print(f"{self.name}: [{label}] Time: {current_time:.2f}s, Memory: {current_memory:.2f} GB, Change: {memory_change:+.2f} GB")
            else:
                print(f"{self.name}: Time: {current_time:.2f}s, Memory: {current_memory:.2f} GB, Change: {memory_change:+.2f} GB")
    
    def plot(self):
        """Plot memory usage over time."""
        import matplotlib.pyplot as plt
        
        if not self.measurements:
            print(f"{self.name}: No measurements to plot.")
            return
        
        times, memories, memory_changes, labels = zip(*self.measurements)
        
        plt.figure(figsize=(10, 6))
        plt.plot(times, memories, 'b-', marker='o')
        plt.title(f"{self.name} - GPU Memory Usage")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory (GB)")
        plt.grid(True)
        plt.show()
    
    def reset(self):
        """Reset the monitor."""
        self.measurements = []
        self.start_time = None
        self.start_memory = None
        print(f"{self.name}: Reset")
        
    def start_continuous_monitoring(self, interval=10, print_msg=False):

        """
        Start continuous memory monitoring in a separate thread.
        
        Args:
            interval: Time between measurements in seconds (default: 10)
            print_msg: Whether to print a message when the monitoring starts (default: False)
        """
        import threading
        import time
        
        def monitor_loop():
            while not self._stop_monitoring:
                self.measure(print_msg=print_msg)
                time.sleep(interval)
        
        self._stop_monitoring = False
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        if print_msg:
            print(f"{self.name}: Started continuous monitoring every {interval} seconds")

    def stop_continuous_monitoring(self):
        """Stop the continuous monitoring thread."""
        self._stop_monitoring = True
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join()
            print(f"{self.name}: Stopped continuous monitoring")
            
    def report(self, print_table=True, save_table=False, save_dir=None):
        """Gather the memory usage and output a table-formatted report"""
        table = Table(title="Memory Usage Report")
        table.add_column("Time", justify="right")
        table.add_column("Memory", justify="right")
        table.add_column("Change", justify="right")
        table.add_column("Label", justify="left")
        
        for time, memory, change, label in self.measurements:
            table.add_row(f"{time:.2f}s", f"{memory:.2f} GB", f"{change:+.2f} GB", label)
            
        if print_table:
            rprint(table)
        
        if save_table:
            if save_dir is None:
                raise ValueError("save_dir must be provided if save_table is True")
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{self.name}_memory_report.txt")
            with open(save_path, "w") as f:
                try:
                    f.write(table.get_string())
                except Exception as e:
                    raise NotImplementedError(f"Error writing to file: {e}")
            
        
        
        

def print_gpu_memory(start_str: str = ""):
    if t.cuda.is_available():
        print(start_str)
        for i in range(t.cuda.device_count()):
            total = t.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
            reserved = t.cuda.memory_reserved(i) / 1024**3
            allocated = t.cuda.memory_allocated(i) / 1024**3
            print(
                f"GPU {i}:",
                f"reserved/allocated/total: {reserved:.2f}/{allocated:.2f}/{total:.2f}",
            )

def shape_of(tensor):
    """
    Simple helper function to print the shape of a tensor.
    
    Args:
        tensor: A PyTorch tensor or any object with a .shape attribute
    
    Example:
        attnout = torch.randn(32, 768)
        s(attnout)  # Output: shape of attnout is torch.Size([32, 768])
    """
    # Get the name of the variable from the caller's frame
    frame = inspect.currentframe().f_back
    calling_line = inspect.getframeinfo(frame).code_context[0].strip()
    # Extract variable name from the function call
    # This looks for s(variable_name) pattern
    import re
    match = re.search(r's\((.*?)\)', calling_line)
    if match:
        var_name = match.group(1).strip()
    else:
        var_name = "tensor"
        
    if hasattr(tensor, 'shape'):
        print(f"Shape of [{var_name}]: {tensor.shape}")
    else:
        print(f"{var_name} has no shape attribute. Type: {type(tensor)}")
        
    pass



if __name__ == "__main__":
    # Testing the memory monitor
    monitor = MemoryMonitor()
    monitor.start()
    # Create a large tensor on CPU
    print("Creating a large tensor on CPU...")
    large_tensor = t.randn(10000, 10000)  # ~400MB tensor
    monitor.measure("After creating large tensor on CPU")
    
    # Move tensor to CUDA if available
    if t.cuda.is_available():
        print("Moving tensor to CUDA...")
        large_tensor = large_tensor.cuda()
        monitor.measure("After moving tensor to CUDA")
        print_gpu_memory("GPU memory after moving tensor to CUDA:")
        
        # Move tensor back to CPU
        print("Moving tensor back to CPU...")
        large_tensor = large_tensor.cpu()
        monitor.measure("After moving tensor back to CPU")
        print_gpu_memory("GPU memory after moving tensor back to CPU:")
        
        # Force CUDA cache clear
        print("Clearing CUDA cache...")
        t.cuda.empty_cache()
        monitor.measure("After clearing CUDA cache")
        print_gpu_memory("GPU memory after clearing cache:")
    else:
        print("CUDA not available, skipping GPU operations")
    
    # Delete the tensor to free memory
    print("Deleting large tensor...")
    del large_tensor
    gc.collect()
    monitor.measure("After deleting large tensor")

    monitor.report(print_table=True, save_table=True, save_dir="output/memory_tables")
