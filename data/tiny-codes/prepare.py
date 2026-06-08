import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

num_proc = 8
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Downloading tiny-codes dataset...")

    # This dataset loads flawlessly because it uses standard, safe Parquet tables
    dataset = load_dataset("nampdn-ai/tiny-codes", split="train")

    # Filter out everything except C++
    print("Filtering for C++ files...")
    cpp_dataset = dataset.filter(lambda x: x["programming_language"] == "C++", num_proc=num_proc)

    # Take a slice for testing (or remove the select call to use all of it)
    sliced_dataset = cpp_dataset.select(range(min(10000, len(cpp_dataset))))

    print(f"Successfully loaded {len(sliced_dataset)} files.")

    # Create train and val splits
    split_dataset = sliced_dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 

    def process(example):
        # Look for the code content in standard column names
        code_content = example.get('response') or example.get('code') or example.get('text')
        if code_content is None:
            return {'ids': [], 'len': 0}
        ids = enc.encode_ordinary(code_content) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing the splits...")
    
    # Drop all textual columns before saving to memory maps
    columns_to_remove = split_dataset['train'].column_names

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