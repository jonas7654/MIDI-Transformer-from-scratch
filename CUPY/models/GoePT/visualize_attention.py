import matplotlib.pyplot as plt
import cupy as cp

def plot_attention_head(attn_matrix, layer=0, head=0, example=0):
    """Plot attention weights for a single head"""
    if isinstance(attn_matrix, cp.ndarray):
        attn_matrix = cp.asnumpy(attn_matrix)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(attn_matrix[example, head], cmap='viridis')
    plt.colorbar()
    plt.title(f"Layer {layer} Head {head}")
    plt.xlabel("Key Position")
    plt.ylabel("Query Position")
    plt.show()

def plot_attention_grid(attention_maps, example=0):
    """Plot all attention heads in a grid"""
    num_layers = len(attention_maps)
    num_heads = attention_maps[0].shape[1]
    
    fig, axes = plt.subplots(num_layers, num_heads, figsize=(num_heads*3, num_layers*3))
    
    for layer_idx, layer_attn in enumerate(attention_maps):
        layer_attn = cp.asnumpy(layer_attn[example])
        for head_idx in range(num_heads):
            ax = axes[layer_idx, head_idx]
            ax.imshow(layer_attn[head_idx], cmap='viridis')
            ax.set_title(f"L{layer_idx} H{head_idx}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()