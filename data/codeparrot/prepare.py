import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset, Dataset

num_proc = 8
enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Connecting to Hugging Face dataset stream...")
    
    # 1. Use the custom dataset script with trust_remote_code=True to bypass the security error
    # 2. Pass the language argument explicitly as defined by the CodeParrot dataset script
    # 3. Use streaming=True to fetch only what we need without downloading the massive full split
    stream = load_dataset(
        "codeparrot/github-code-clean", 
        trust_remote_code=True, 
        languages=["C++"], 
        split="train",
        streaming=True
    )

    # Fetch a specific number of C++ files directly from the stream.
    # The full C++ dataset is ~7.3 million files. 5% would be ~369,000 files.
    # We are pulling 10,000 here for testing purposes. Adjust as needed for your nanoMoE target.
    print("Downloading dataset slice into memory...")
    sliced_data = list(stream.take(10000)) 
    
    # Convert back to a standard Dataset object for normal map/tokenize processing
    sliced_dataset = Dataset.from_list(sliced_data)

    print(f"Successfully loaded {len(sliced_dataset)} files.")

    # Create train and val splits
    split_dataset = sliced_dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 

    def process(example):
        # CodeParrot standardizes on 'code' for the source text
        code_content = example.get('code') or example.get('text')
        if code_content is None:
            return {'ids': [], 'len': 0}
        ids = enc.encode_ordinary(code_content) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    print("Tokenizing the splits...")
    
    # Dynamically find columns to remove to avoid mapping crashes
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