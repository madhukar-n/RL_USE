import os
import matplotlib
# Force a headless backend to prevent errors on servers/SSH
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_augmix_views(images_list):
    """
    Saves a grid of AugMix views directly to disk as a PNG file.
    Does not open popup windows.
    """
    output_dir = "./utils/"
    num_views = len(images_list)
    if num_views == 0:
        return

    # Standard CLIP denormalization constants
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

    # Limit maximum displayed views to 9 columns to fit screens comfortably
    max_display = min(num_views, 9)
    fig, axes = plt.subplots(1, max_display, figsize=(max_display * 3, 3))
    
    if max_display == 1:
        axes = [axes]

    for i in range(max_display):
        img_tensor = images_list[i].detach().cpu()
        
        # Squeeze out batch dimension if present
        if len(img_tensor.shape) == 4:
            img_tensor = img_tensor.squeeze(0)
            
        # Denormalize pixels back to RGB range [0, 1]
        img_tensor = img_tensor * std + mean
        img_np = img_tensor.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        axes[i].imshow(img_np)
        if i == 0:
            axes[i].set_title("Base View", fontsize=10, fontweight='bold')
        else:
            axes[i].set_title(f"AugMix {i}", fontsize=9)
        axes[i].axis('off')

    plt.tight_layout()
    
    # Create a dedicated visual directory inside your existing output path
    viz_dir = os.path.join(output_dir, 'augmented_visuals')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Save target file
    save_path = os.path.join(viz_dir, f'iter.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig) # Free memory safely
    # print(f"==> Saved augmentation visualization grid to: {save_path}")
