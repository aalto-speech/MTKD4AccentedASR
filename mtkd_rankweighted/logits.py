from tqdm import tqdm
import os
import glob
import numpy as np
from datasets import load_from_disk


def add_avg_logits(example, avg_logits):
    """
    Adds precomputed average logits to a dataset example based on its audio path.

    This function:
    - Looks up the average logits corresponding to the example's audio file path.
    - If no matching entry is found in the `avg_logits` dictionary, assigns a default zero array.
    - Appends the logits under a new key `"avg_logits"` in the example.

    Args:
        example (dict): A single dataset example that must include the key "audio_path".
        avg_logits (dict): A dictionary mapping audio file paths to their corresponding average logits 
                           (e.g., {audio_path: np.ndarray}).

    Returns:
        dict: The input example, augmented with a new key "avg_logits" containing the matched or default logits.
    """
    
    example["avg_logits"] = avg_logits.get(example["audio_path"], np.zeros((1,)))  # Default to an empty array if not found
    return example





def load_dataset_logits(accent, saved_logits_path, excluded_models):
    """
    Loads datasets containing model output logits for a specific accent, 
    excluding certain models from the loading process.

    This function:
    - Searches through directories matching the `saved_logits_path` pattern.
    - Filters out any paths that contain substrings listed in `excluded_models`.
    - For each valid path, loads a dataset from the subdirectory `tested_on_{accent}`.

    Args:
        accent (str): The name of the accent to filter datasets (used to locate subdirectories).
        saved_logits_path (str): A glob pattern for locating saved model directories.
                                 Example: "logits_outputs/*" or a specific path.
        excluded_models (list): A list of substrings; if any of these appear in a model path, 
                                that model is excluded from loading.

    Returns:
        list: A list of loaded datasets (e.g., Hugging Face `Dataset` objects), 
              one for each included model.
    """
    
    return [
        load_from_disk(os.path.join(model, f"tested_on_{accent}"))
        for model in glob.glob(saved_logits_path)
        if not any(excluded in model for excluded in excluded_models)
    ]
    
    
    
    
    
def compute_avg_logits(dataset_logits):
    """
    Computes the average (summed) logits across multiple models for each audio example.

    This function:
    - Iterates through each audio example in the datasets (assumes all datasets are aligned in order and size).
    - Extracts the logits for the same index across all model outputs.
    - Verifies that the `audio_path` matches across all models for that index.
    - Sums the logits element-wise across models.
    - Stores the result in a dictionary using the `audio_path` as the key.

    Notes:
    - This version stores the raw summed logits (no averaging or softmax is applied).
    - You can uncomment other blocks in the function to perform averaging or softmax normalization if needed.

    Args:
        dataset_logits (list): A list of datasets (e.g., Hugging Face `Dataset` objects) 
                               where each contains model outputs including:
                               - "logits": A 2D numpy array (e.g., shape [time, classes]).
                               - "audio_path": A unique identifier for the audio file.

    Returns:
        dict: A dictionary mapping `audio_path` (str) to the summed logits (np.ndarray).
    """
    
    avg_logits = {}
    num_models = len(dataset_logits)
    
    for idx in tqdm(range(len(dataset_logits[0]))):
        logits_list = [np.array(dataset_logits[i][idx]['logits']) for i in range(num_models)]  # Shape: (10, 128, 32)

        if all(dataset_logits[i][idx]['audio_path'] == dataset_logits[0][idx]['audio_path'] for i in range(num_models)):
            summed_logits = np.sum(logits_list, axis=0)  # Summing element-wise (Result: 128x32)

            # # Apply softmax normalization (normalize along last axis)
            # exp_logits = np.exp(summed_logits - np.max(summed_logits, axis=-1, keepdims=True))  # Avoid overflow
            # softmax_logits = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # Softmax normalization
            # avg_logits[dataset_logits[0][idx]['audio_path']] = softmax_logits  # Store result
            
            # # Normalize by dividing by 10
            # normalized_logits = summed_logits / 10.0
            # avg_logits[dataset_logits[0][idx]['audio_path']] = normalized_logits
            
            # raw values
            avg_logits[dataset_logits[0][idx]['audio_path']] = summed_logits
            
        else:
            print("Size mismatch issue in compute_avg_logits fn")
    
    return avg_logits




def merge_logits(dataset_list):
    """
    Merge logits from multiple datasets (e.g., teacher models) into one dataset
    by adding new keys like 'logits1', 'logits2', ..., 'logitsN' to each example.

    Args:
        dataset_list (List[datasets.Dataset]): List of Hugging Face Dataset objects,
            each with a 'logits' column.

    Returns:
        datasets.Dataset: A dataset where each sample contains logits from all sources.
    """
    
    def map_fn(batch):
        batch["audio_path"] = batch["audio_path"][0]
        for idx, ds in enumerate(dataset_list):
            batch[f"logits{idx+1}"] = ds["logits"]
        return batch

    merged_dataset = dataset_list[0].map(map_fn, batched=True, remove_columns=["logits"])
    
    return merged_dataset




def merge_multiple_logits_datasets(dataset_list):
    """
    Merge logits from multiple aligned Hugging Face Datasets into a single dataset.

    Args:
        dataset_list (List[datasets.Dataset]): A list of datasets, each containing a 'logits' column.
                                               All datasets must be aligned (same order and length).

    Returns:
        datasets.Dataset: A dataset containing the original data and additional
                          'logits1', 'logits2', ..., 'logitsN' columns.
    """
    
    # Base dataset without logits
    base_dataset = dataset_list[0].remove_columns(["logits"])

    def merge_logits(batch, indices):
        merged_logits = {
            f"logits{i+1}": [dataset_list[i]["logits"][idx] for idx in indices] 
            for i in range(len(dataset_list))
        }
        return merged_logits

    # Use batched=True to process multiple rows at once
    merged_dataset = base_dataset.map(merge_logits, with_indices=True, batched=True, batch_size=512)

    return merged_dataset