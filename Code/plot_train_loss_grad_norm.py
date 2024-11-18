import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
i = 0
df = pd.read_csv(f"log_histories/log_{i}.csv")

# Create the figure and axis object with two vertically aligned subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# Plot Loss over Steps
axs[0].plot(df['step'], df['loss'], color='blue', label='Loss')
axs[0].set_xlabel('Step')
axs[0].set_ylabel('Loss')
axs[0].grid(True)
axs[0].legend()

# Plot Gradient Norm over Steps
axs[1].plot(df['step'], df['grad_norm'], color='red', label='Gradient Norm')
axs[1].set_xlabel('Step')
axs[1].set_ylabel('Gradient Norm')
axs[1].grid(True)
axs[1].legend()

# Adjust layout and make the figure tight
plt.tight_layout()

# Save the figure
plt.savefig(f'metrics_vs_step_{i}.png')
plt.show()
