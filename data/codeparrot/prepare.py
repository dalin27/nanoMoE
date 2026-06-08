# saves the codeparrot C++ dataset subset to a binary file for training.
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets

# number of workers in .map() call
num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    print("Loading C++ dataset shards via stable Parquet storage branch...")
    
    # 1. Bypasses code execution limits by pointing directly to the automated Parquet shard
    data_files = [
        "https://huggingface.co/datasets/codeparrot/github-code/resolve/refs%2Fconvert%2Fparquet/C%2B%2B-all/train/0000.parquet"
    ]
    
    # Using 'parquet' engine stops the library from looking for 'github-code.py'
    dataset = load_dataset("parquet", data_files=data_files)

    # 2. Slice to 5% of the train data as requested in your original script
    # This gives you plenty of data for 100 steps without needing to map the entire shard
    sliced_dataset = dataset["train"].select(range(len(dataset["train"]) // 20))

    # Create train and val splits
    split_dataset = sliced_dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val

    # Tokenizer processing function adapted for code strings
    def process(example):
        # The text column in codeparrot/github-code is named 'code'
        ids = enc.encode_ordinary(example['code']) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['code', 'repo_name', 'path', 'language', 'license', 'size'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Make sure output data location aligns dynamically
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'cpp_dataset')
    os.makedirs(output_dir, exist_ok=True)

    # concatenate all the ids in each dataset into one large file we can use for training
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