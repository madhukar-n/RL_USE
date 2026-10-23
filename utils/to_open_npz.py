import numpy as np
import matplotlib
# Use 'Agg' backend for headless server environments (SSH)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 1. Load the data (Update this path to your actual saved .npz file)
file_path = '../temp/RN50/seed_0/se/tracking_data_A.npz' 
data = np.load(file_path)

all_entropies = data['entropies']
is_correct_list = data['is_correct']
pseudo_accuracies = data['pseudo_accuracies']
data.close()

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# --- Plot 1: Pseudo-Label Accuracy Degradation ---
steps = np.arange(1, len(pseudo_accuracies) + 1)
ax1.plot(steps, pseudo_accuracies, color='darkred', linewidth=2, label='Pseudo-Label Accuracy')
ax1.set_title('Pseudo-Label Accuracy Degradation Over Time')
ax1.set_xlabel('Adaptation Step (Number of Samples)')
ax1.set_ylabel('Cumulative Accuracy')
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()

# --- Plot 2: Entropy vs. Correctness Scatter Plot ---
incorrect_mask = ~is_correct_list
correct_mask = is_correct_list

# Plot incorrect predictions (Red 'x')
ax2.scatter(np.where(incorrect_mask)[0], all_entropies[incorrect_mask], 
            color='red', alpha=0.6, label='Incorrect Pseudo-Label', marker='x')

# Plot correct predictions (Green 'o')
ax2.scatter(np.where(correct_mask)[0], all_entropies[correct_mask], 
            color='green', alpha=0.4, label='Correct Pseudo-Label', marker='o')

# Add a visual threshold line (adjust 0.5 based on your actual entropy scale)
ax2.axhline(y=0.5, color='black', linestyle='--', label='Theoretical Entropy Filter')

ax2.set_title('Ensemble Entropy vs. Prediction Correctness')
ax2.set_xlabel('Sample Index')
ax2.set_ylabel('Entropy Score')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend()

# Save the figure to a PNG file instead of displaying it
plt.tight_layout()
save_path = 'confirmation_bias_plots.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Plot saved successfully to {save_path}")
