import matplotlib.pyplot as plt
import pandas as pd

# Example data (for demonstration, you should load your own dataset)
data = pd.read_csv("/home/ubuntu/abdes/glora_experiments/some_inference/log.csv")

# Convert data to a DataFrame
df = pd.DataFrame(data)

# Extract training and validation loss
epochs = df['step']
train_loss = df['loss']
eval_steps = df['step'][df['eval_loss'].notna()]
eval_loss = df['eval_loss'].dropna()


# Plotting both training and validation loss on the same figure
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss', color='blue', linewidth=1)
plt.scatter(eval_steps, eval_loss, label='Validation Loss', color='red', s=30, marker='o')


# Add labels and title
plt.xlabel("Step")
plt.ylabel("Loss")
plt.legend(loc="best")

# Show grid and plot
plt.grid(True)
plt.show()
plt.savefig('loss.png')
