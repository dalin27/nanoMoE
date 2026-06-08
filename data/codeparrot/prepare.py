import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset
from huggingface_hub import HfFileSystem

num_proc = 8
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Resolving exact file paths using HfFileSystem...")

    fs = HfFileSystem()
    
    # 1. Ask the Hub API directly for the list of files. 
    # This bypasses the internal wildcard bugs with special characters.
    repo_path = "datasets/codeparrot/github-code@refs/convert/parquet/C++-all/**/*.parquet"
    parquet_files = fs.glob(repo_path)
    
    if not parquet_files:
        raise FileNotFoundError(f"Could not find any parquet files at {repo_path}")

    # 2. Build explicit, safe URIs for the dataset builder
    data_files = []
    for f in parquet_files:
        if "@" not in f:
            path_in_repo = f.split("codeparrot/github-code/", 1)[-1]
        else:
            path_in_repo = f.split("@refs/convert/parquet/", 1)[-1]
            
        # Crucial fix: URL-encode plus symbols so they are not parsed as spaces
        safe_path = path_in_repo.replace('+', '%2B')
        data_files.append(f"hf://datasets/codeparrot/github-code@refs/convert/parquet/{safe_path}")

    print(f"Found {len(data_files)} Parquet shards. Connecting stream...")

    # 3. Pass the explicit list of resolved files to the pure Parquet builder
    stream = load_dataset(
        "parquet",
        data_files={"train": data_files},
        split="train",
        streaming=True
    )

    print("Downloading dataset slice into memory...")
    sliced_data = list(stream.take(10000)) 
    
    sliced_dataset = Dataset.from_list(sliced_data)

    print(f"Successfully loaded {len(sliced_dataset)} files.")

    # Create train and val splits
    split_dataset = sliced_dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 

    def process(example):
        code_content = example.get('code') or example.get('text')
        if code_content is None:
            return {'ids': [], 'len': 0}
        ids = enc.encode_ordinary(code_content) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing the splits...")
    
    columns_to_remove = [col for col in split_dataset['train'].column_names if col not in ['ids', 'len']]

    tokenized = split_dataset.map(
        process,
        remove_columns=columns_to_remove,
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'cpp_dataset')
    os.makedirs(output_dir, exist_ok=True)

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16 
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        total_batches = 1024

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            if len(batch['ids']) > 0:
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
        arr.flush()
        print(f"Saved {filename} with {arr_len} tokens.")