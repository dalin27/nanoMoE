import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset 
from huggingface_hub import HfFileSystem

num_proc = 8
num_proc_load_dataset = num_proc

# MUST remain gpt2 to match OpenWebText embeddings
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Fetching C++ Parquet file paths from Hugging Face Hub...")
    
    # 1. Initialize Hugging Face virtual file system to inspect the parquet branch
    fs = HfFileSystem()
    
    # List all Parquet shards inside the C++ subset folder
    # This automatically finds train-00000-of-01126.parquet, train-00001..., etc.
    cpp_files = fs.glob("datasets/codeparrot/github-code@refs/convert/parquet/C++-all/*.parquet")
    
    # Convert paths back to standard hf:// URLs that load_dataset natively understands
    data_files = [f"hf://{file}" for file in cpp_files]
    
    # Sort them to keep order consistent
    data_files = sorted(data_files)
    
    print(f"Found {len(data_files)} Parquet chunks. Loading the 5% slice...")

    # 2. Load natively using the 'parquet' engine, bypassing code execution
    dataset = load_dataset("parquet", data_files=data_files, split="train[:5%]")
    print(f"Successfully loaded dataset slice with {len(dataset)} documents.")

    # Create train and val splits
    split_dataset = dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')

    def process(example):
        # The text column in codeparrot/github-code is 'code'
        ids = enc.encode_ordinary(example['code']) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['code', 'repo_name', 'path', 'language', 'license', 'size'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Ensure the output directory matches the dynamic router logic
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
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Saved {filename} with {arr_len} tokens.")