import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 

num_proc = 8
num_proc_load_dataset = num_proc

# MUST remain gpt2 to match OpenWebText embeddings
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Loading C++ dataset directly via Parquet backend...")
    
    # Define the remote Parquet directory structure managed by Hugging Face
    base_url = "https://huggingface.co/datasets/codeparrot/github-code/resolve/refs%2Fconvert%2Fparquet/C%2B%2B-all"
    
    # There is 1 primary shard for C++-all. We point directly to it.
    data_files = {
        "train": f"{base_url}/train-00000-of-00001.parquet"
    }
    
    # Load via the native, safe 'parquet' reader to avoid the blocked .py script
    dataset = load_dataset("parquet", data_files=data_files, split="train[:5%]")

    print(f"Loaded dataset size: {len(dataset)} files.")

    # Create train and val splits
    split_dataset = dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')