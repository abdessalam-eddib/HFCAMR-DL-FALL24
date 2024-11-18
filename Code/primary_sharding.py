from datasets import load_dataset, Dataset
import numpy as np
from collections import defaultdict

def create_shards(dataset, num_shards=20, label_column='label'):
    """
    Split a classification dataset into similar size shards maintaining class distribution.
    
    Args:
        dataset: HuggingFace dataset object
        num_shards: Number of shards to create
        label_column: Name of the column containing labels
        
    Returns:
        List of dataset shards (each shard is a Hugging Face Dataset object)
    """
    # Convert dataset to list for easier manipulation
    dataset_list = list(dataset)
    
    # Group samples by label
    label_groups = defaultdict(list)
    for idx, sample in enumerate(dataset_list):
        label_groups[sample[label_column]].append(idx)
    
    # Calculate samples per shard for each class
    samples_per_shard = {
        label: len(indices) // num_shards 
        for label, indices in label_groups.items()
    }
    
    # Create shards
    shards = [[] for _ in range(num_shards)]
    
    # Distribute samples to shards
    for label, indices in label_groups.items():
        # Shuffle indices for randomization
        np.random.shuffle(indices)
        
        # Calculate samples per shard for this class
        shard_size = samples_per_shard[label]
        
        # Distribute indices across shards
        for shard_idx in range(num_shards):
            start_idx = shard_idx * shard_size
            end_idx = start_idx + shard_size
            
            # Get indices for this shard
            if shard_idx == num_shards - 1:
                # Last shard gets remaining samples
                shard_indices = indices[start_idx:]
            else:
                shard_indices = indices[start_idx:end_idx]
            
            # Add samples to shard
            shards[shard_idx].extend(shard_indices)
    
    # Convert index lists back to HuggingFace Dataset objects
    final_shards = []
    for shard_indices in shards:
        shard_data = [dataset_list[i] for i in sorted(shard_indices)]
        final_shards.append(Dataset.from_dict({key: [item[key] for item in shard_data] for key in shard_data[0].keys()}))
        
    return final_shards

def verify_shard_balance(shards, label_column='label'):
    """
    Verify that shards maintain similar class distributions.
    
    Args:
        shards: List of dataset shards
        label_column: Name of the column containing labels
        
    Returns:
        dict: Statistics about class distribution in shards
    """
    stats = defaultdict(list)
    
    for shard_idx, shard in enumerate(shards):
        # Count labels in shard
        label_counts = defaultdict(int)
        for sample in shard:
            label_counts[sample[label_column]] += 1
            
        # Calculate distribution
        total_samples = len(shard)
        distribution = {
            label: count / total_samples 
            for label, count in label_counts.items()
        }
        
        # Store stats
        stats['shard_sizes'].append(total_samples)
        for label, ratio in distribution.items():
            stats[f'class_{label}_ratio'].append(ratio)
    
    return dict(stats)

# Example usage:
if __name__ == "__main__":
    # Load your dataset
    dataset = load_dataset("ad6398/nyu-dl-teach-maths-comp", split="train")
    
    # Create balanced shards
    shards = create_shards(dataset, num_shards=20, label_column='is_correct')

    for i, shard in enumerate(shards):
        # save the shard to disk
        shard.save_to_disk(f"train_sets/shard_{i}")
    
    # Verify balance
    stats = verify_shard_balance(shards, label_column='is_correct')
    
    # Print statistics
    print(f"Number of shards: {len(shards)}")
    print(f"Shard sizes: min={min(stats['shard_sizes'])}, max={max(stats['shard_sizes'])}")
    
    for label in [key for key in stats.keys() if key.startswith('class_')]:
        ratios = stats[label]
        print(f"{label} distribution: min={min(ratios):.3f}, max={max(ratios):.3f}, "
              f"std={np.std(ratios):.3f}")
