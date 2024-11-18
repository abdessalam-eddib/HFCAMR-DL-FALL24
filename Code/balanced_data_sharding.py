from datasets import load_dataset, concatenate_datasets
import pandas as pd
import logging
from typing import Optional, Tuple
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_balance_dataset(
    dataset_name: str,
    num_samples: int = 30000,
    split: str = 'train',
    seed: int = 42
) -> Optional[concatenate_datasets]:
    """
    Load a dataset and balance it by selecting equal numbers of True and False samples.
    
    Args:
        dataset_name: Name of the dataset to load
        num_samples: Number of samples to select for each class
        split: Dataset split to use
        seed: Random seed for reproducibility
    
    Returns:
        Combined dataset with balanced True and False samples
    """
    try:
        # Load the dataset
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)
        
        # Verify 'is_correct' column exists
        if 'is_correct' not in dataset.features:
            raise ValueError("Dataset does not contain 'is_correct' column")
        
        # Filter True and False samples
        logger.info("Filtering samples by is_correct value")
        true_samples = dataset.filter(lambda x: x['is_correct'] is True)
        false_samples = dataset.filter(lambda x: x['is_correct'] is False)
        
        # Check if we have enough samples
        min_samples_available = min(len(true_samples), len(false_samples))
        if min_samples_available < num_samples:
            logger.warning(
                f"Requested {num_samples} samples per class but only {min_samples_available} "
                "available. Adjusting sample size."
            )
            num_samples = min_samples_available
        
        # Shuffle and select samples
        logger.info(f"Selecting {num_samples} samples from each class")
        random.seed(seed)
        true_samples_shard = true_samples.shuffle(seed=seed).select(range(num_samples))
        false_samples_shard = false_samples.shuffle(seed=seed).select(range(num_samples))

        # Validation set out of the remaining samples
        true_samples_val = true_samples.select(range(num_samples, len(true_samples)))
        false_samples_val = false_samples.select(range(num_samples, len(false_samples)))

        # Only keep 2000 samples for validation
        true_samples_val = true_samples_val.select(range(500))
        false_samples_val = false_samples_val.select(range(500))

        # Combine the validation sets
        logger.info("Combining validation datasets")
        val_dataset = concatenate_datasets([true_samples_val, false_samples_val])
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Combine the shards
        logger.info("Combining balanced datasets")
        combined_dataset = concatenate_datasets([true_samples_shard, false_samples_shard])
        
        # Shuffle the combined dataset
        combined_dataset = combined_dataset.shuffle(seed=seed)
        
        logger.info(f"Final dataset size: {len(combined_dataset)}")
        return combined_dataset, val_dataset
    
    except Exception as e:
        logger.error(f"Error processing dataset: {str(e)}")
        raise

def save_dataset(dataset, output_path: str, format: str = 'arrow') -> None:
    """
    Save the dataset to disk in the specified format.
    
    Args:
        dataset: The dataset to save
        output_path: Path where to save the dataset
        format: Format to save the dataset in ('arrow', 'csv', 'json', 'parquet')
    """
    try:
        logger.info(f"Saving dataset to: {output_path}")
        if format == 'arrow':
            dataset.save_to_disk(output_path)
        elif format == 'csv':
            df = dataset.to_pandas()
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df = dataset.to_pandas()
            df.to_json(output_path, orient='records')
        elif format == 'parquet':
            df = dataset.to_pandas()
            df.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info("Dataset saved successfully")
    
    except Exception as e:
        logger.error(f"Error saving dataset: {str(e)}")
        raise

def main():
    # Example usage
    try:
        # Configure these parameters as needed
        DATASET_NAME = 'ad6398/nyu-dl-teach-maths-comp'
        OUTPUT_PATH_VAL = './val_set_final'
        NUM_SAMPLES = 41000
        SAVE_FORMAT = 'arrow'
        
        # Load and balance the dataset
        balanced_dataset, val_dataset = load_and_balance_dataset(
            dataset_name=DATASET_NAME,
            num_samples=NUM_SAMPLES,
            seed=42, 
        )
        
        # Save the dataset
        if balanced_dataset is not None:
            save_dataset(val_dataset, OUTPUT_PATH_VAL, format=SAVE_FORMAT)
    
    except Exception as e:
        logger.error(f"Failed to process dataset: {str(e)}")
        raise

if __name__ == "__main__":
    main()