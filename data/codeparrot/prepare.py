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
    # Load only the C++ subset of the codeparrot dataset.
    # We use a slice ['train[:5%]'] because you only need enough data for 100 steps.
    # 5% is still hundreds of thousands of documents, which is more than enough.
    dataset = load_dataset("codeparrot/github-code", languages=["C++"], split="train[:5%]", num_proc=num_proc_load_dataset)

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