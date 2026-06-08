import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

num_proc = 8
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Bypassing dataset scripts and connecting directly to Parquet shards...")

    # Force the pure parquet builder and point it directly to the hidden auto-converted branch
    # This completely ignores the banned python execution files.
    stream = load_dataset(
        "parquet",
        data_files="hf://datasets/codeparrot/github-code@refs/convert/parquet/C++-all/train/*.parquet",
        split="train",
        streaming=True
    )

    # Fetch a specific number of C++ files directly from the stream.
    # Adjust this number based on your exact nanoMoE scale requirements.
    print("Downloading dataset slice into memory...")
    sliced_data = list(stream.take(10000)) 
    
    # Convert back to a standard Dataset object for mapping
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
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            if len(batch['ids']) > 0:
                arr_batch = np.concatenate(batch['ids'])
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
        arr.flush()
        print(f"Saved {filename} with {arr_len} tokens.")