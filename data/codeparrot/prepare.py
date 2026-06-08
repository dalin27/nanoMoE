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
    print("Loading dataset in streaming mode...")
    # 1. Use streaming=True to completely bypass the deprecated dataset script block
    remote_dataset = load_dataset("codeparrot/github-code", languages=["C++"], split="train", streaming=True)

    # 2. Pull the exact number of samples needed for your 5% slice safely
    # The total C++ files is ~7.3 Million. 5% is roughly 369,000 files.
    num_samples = 369000 
    print(f"Taking {num_samples} samples from the C++ stream...")
    
    # Take the slice and convert the stream into a local standard Dataset object
    dataset = remote_dataset.take(num_samples)
    dataset = list(dataset)
    from datasets import Dataset
    dataset = Dataset.from_list(dataset)

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