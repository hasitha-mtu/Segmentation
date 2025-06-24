import tensorflow as tf

def estimate_model_memory_usage(model, batch_size, dtype=tf.float32):
    # Get parameter size
    param_count = model.count_params()
    bytes_per_param = tf.dtypes.as_dtype(dtype).size  # e.g., 4 for float32

    # Weights
    param_mem = param_count * bytes_per_param

    # Gradients (same as weights)
    grad_mem = param_mem

    # Optimizer states (Adam: 2x param size)
    opt_mem = 2 * param_mem

    # Activation memory (very rough estimate)
    # Calculate memory for all intermediate feature maps in the model
    activation_mem = 0
    for layer in model.layers:
        try:
            output_shape = layer.output_shape
            if isinstance(output_shape, list):
                output_shape = output_shape[0]
            if None in output_shape:
                # Replace None with batch size (usually in batch dimension)
                output_shape = [batch_size if dim is None else dim for dim in output_shape]

            # Ensure valid shape
            if not all(isinstance(dim, int) and dim > 0 for dim in output_shape):
                continue  # Skip invalid shapes

            num_elements = 1
            for dim in output_shape:
                num_elements *= dim
            activation_mem += num_elements * bytes_per_param

        except Exception as e:
            # Optional: print debug info
            print(f"Skipped layer: {layer.name}, reason: {e}")
            continue

    # Total memory
    total_mem_bytes = param_mem + grad_mem + opt_mem + activation_mem
    total_mem_gb = total_mem_bytes / (1024 ** 3)

    print(f"Estimated model memory usage (batch size {batch_size}):")
    print(f"- Parameters       : {param_mem / (1024 ** 2):.2f} MB")
    print(f"- Gradients        : {grad_mem / (1024 ** 2):.2f} MB")
    print(f"- Optimizer states : {opt_mem / (1024 ** 2):.2f} MB")
    print(f"- Activations      : {activation_mem / (1024 ** 2):.2f} MB")
    print(f"≈ Total            : {total_mem_gb:.2f} GB (approx)")

if __name__=='__main__':
    from tensorflow.keras.applications import MobileNetV2
    model = MobileNetV2(input_shape=(512, 512, 3), weights=None, include_top=False)
    estimate_model_memory_usage(model, batch_size=4)