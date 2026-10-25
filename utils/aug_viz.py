import matplotlib.pyplot as plt
import torch
import numpy as np

def visualize_augmix_views(sample_image, data_transform):
    """
    Passes a single image through your AugMixAugmenter 
    and plots all generated views side by side.
    """
    # 1. Generate the multi-view augmentations
    # Depending on your codebase, it returns a list of Tensors or a stacked Tensor
    views = data_transform(sample_image)
    
    # 2. Determine the total number of views generated
    num_views = len(views)
    
    # 3. Setup the Matplotlib grid
    fig, axes = plt.subplots(1, num_views, figsize=(num_views * 3, 3))
    
    # Handle single view edge-case safely for indexing
    if num_views == 1:
        axes = [axes]
        
    for i, view in enumerate(views):
        # Convert to Tensor if it's a PIL image, or copy if it's already a Tensor
        if not isinstance(view, torch.Tensor):
            from torchvision.transforms import functional as F
            img_tensor = F.to_tensor(view)
        else:
            img_tensor = view.detach().cpu()
            
        # Denormalize if your 'preprocess' transform includes Normalize((mean), (std))
        # Modify these values if you use Custom / ImageNet scales
        if img_tensor.shape[0] == 3:
            # Example for standard ImageNet denormalization (uncomment if needed):
            # mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            # std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            # img_tensor = img_tensor * std + mean
            
            # Convert CHW Tensor to HWC Numpy array for Matplotlib
            img_np = img_tensor.permute(1, 2, 0).numpy()
            img_np = np.clip(img_np, 0, 1) # Keep pixels in stable [0, 1] range
        else:
            img_np = img_tensor.numpy()

        # 4. Render the plot
        axes[i].imshow(img_np)
        axes[i].set_title(f"View {i+1}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

# --- Execution Example ---
# raw_img = Image.open("test.jpg").convert("RGB")
# visualize_augmix_views(raw_img, data_transform)
