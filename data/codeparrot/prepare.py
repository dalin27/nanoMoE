import os
from tqdm import tqdm
import numpy as np
import tiktoken
import urllib.request
from datasets import load_dataset 

num_proc = 8
num_proc_load_dataset = num_proc

enc = tiktoken.get_encoding("gpt2")

if __name__ == '__main__':
    # 1. Download a stable, direct Parquet shard from the Hub over basic HTTPS
    # This URL points to a clean, mirrored chunk of the github-code C++ dataset
    url = "https://huggingface.co/datasets/codeparrot/github-code/resolve/main/data/C%2B%2B-all-train.parquet"
    local_parquet = os.path.join(os.path.dirname(__file__), "cpp_shard.parquet")
    
    if not os.path.exists(local_parquet):
        print(f"Downloading C++ data shard directly to {local_parquet}...")
        # Custom opener to handle Hugging Face redirects gracefully
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(url, local_parquet)
        print("Download complete.")
    else:
        print("Using existing local C++ Parquet shard.")

    # 2. Load the local file natively (exactly like your openwebtext fix)
    print("Loading local Parquet data...")
    dataset = load_dataset("parquet", data_files=local_parquet)

    # 3. Take a 5% slice of the shard
    sliced_dataset = dataset["train"].select(range(len(dataset["train"]) // 20))

    # Create train and val splits
    split_dataset = sliced_dataset.train_test_split(test_size=0.005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') 

    def process(example):
        # The text column in codeparrot/github-code is 'code'
        if example['code'] is None:
            return {'ids': [], 'len': 0}
        ids = enc.encode_ordinary(example['code']) 
        ids.append(enc.eot_token) 
        out = {'ids': ids, 'len': len(ids)}
        return out

    # Tokenize the dataset
    print("Tokenizing the splits...")
    tokenized = split_dataset.map(
        process,
        remove_columns=['code', 'repo_name', 'path', 'language', 'license', 'size'],
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
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        print(f"Saved {filename} with {arr_len} tokens.")