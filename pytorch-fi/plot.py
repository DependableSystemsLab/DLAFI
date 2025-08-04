import matplotlib.pyplot as plt

def plot_max_values(max_values):
    """
    Plots a histogram of the maximum values of different channels for each convolutional layer.

    Args:
    - max_values (dict): A dictionary where keys are layer identifiers (names or objects) and
                         values are tensors of maximum values for each output channel of the layer.
    """
    # Create a figure with subplots
    num_layers = len(max_values)
    fig, axs = plt.subplots(num_layers, 1, figsize=(10, num_layers * 5))
    
    if num_layers == 1:  # Adjust if there's only one layer to avoid indexing issues
        axs = [axs]
    
    for ax, (layer, values) in zip(axs, max_values.items()):
        # Use the layer name if it's available, otherwise use its index
        layer_name = str(layer) if isinstance(layer, str) else f"Layer {list(max_values.keys()).index(layer)}"
        ax.hist(values.cpu().numpy(), bins=20)  # Convert to numpy array and plot histogram
        ax.set_title(f"Max Values Distribution for {layer_name}")
        ax.set_xlabel("Max Value")
        ax.set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()

# Example usage assuming max_values is your dictionary of max values for each layer
# plot_max_values(max_values)

def plot_max_values_2d(max_values):
    """
    Plots a histogram of the maximum values of different channels for each convolutional layer
    in a 3x6 grid layout.

    Args:
    - max_values (dict): A dictionary where keys are layer identifiers (names or objects) and
                         values are tensors of maximum values for each output channel of the layer.
    """
    # Create a figure with subplots in a 3x6 grid
    fig, axs = plt.subplots(4, 6, figsize=(20, 10))  # Adjust figsize as needed
    axs = axs.flatten()  # Flatten the array to easily iterate over it
    
    for ax, (layer, values) in zip(axs, max_values.items()):
        layer_name = str(layer) if isinstance(layer, str) else f"Layer {list(max_values.keys()).index(layer)}"
        ax.hist(values.cpu().numpy(), bins=20)  # Convert to numpy array and plot histogram
        ax.set_title(f"Max Values for {layer_name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    
    # Hide unused subplots if there are any
    for i in range(len(max_values), len(axs)):
        axs[i].axis('off')

    plt.tight_layout()
    plt.show()
