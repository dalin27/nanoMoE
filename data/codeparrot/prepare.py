import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 
from huggingface_hub import hf_hub_download

num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Downloading C++ data shard using official huggingface_hub API...")
    
    # Use the consistent repo ID you intend to target
    target_repo = "codeparrot/github-code-clean"
    
    try:
        # 1. Attempt to cache the specific parquet shard locally
        local_parquet = hf_hub_download(
            repo_id=target_repo,
            filename="data/C++-all-train.parquet", # Adjusted to match github-code-clean paths if applicable
            repo_type="dataset"
        )
        print(f"File resolved successfully at: {local_parquet}")
        print("Loading local Parquet data into memory...")
        dataset = load_dataset("parquet", data_files=local_parquet)
        
    except Exception as e:
        print(f"Hub download failed: {e}")
        print("Falling back to loading directly via datasets API...")
        # Emergency backup: Let the datasets library handle the remote fetch natively
        # mapping directly to the repository and subset/split
        dataset = load_dataset(target_repo, data_files={"train": "data/C++-all-train.parquet"})

    # 3. Take a 5% slice of the data shard
    dataset_split = dataset["train"]
    sliced_dataset = dataset_split.select(range(len(dataset_split) // 20))

    # Create train and val splits
    split_dataset = sliced_dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 

    def process(example):
        # Fallback check for both 'code' or 'text' column names depending on the exact dataset variant
        code_content = example.get('code') or example.get('text')
        if code_content is None:
            return {'ids': [], 'len': 0}
        ids = enc.encode_ordinary(code_content) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    print("Tokenizing the splits...")
    
    # Dynamically find columns to remove so it doesn't crash if a column is missing
    columns_to_remove = [col for col in split_dataset['train'].column_names if col not in ['ids', 'len']]

    tokenized = split_dataset.map(
        process,
        remove_columns=columns_to_remove,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Ensure the output directory matches your dynamic router logic
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'cpp_dataset')
    os.makedirs(output_dir, exist_ok=True)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            if len(batch['ids']) > 0:
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
        arr.flush()
        print(f"Saved {filename} with {arr_len} tokens.")