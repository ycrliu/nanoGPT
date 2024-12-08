import os
import requests
import tiktoken
import numpy as np
import random
import argparse


def prep(filename, add_canaries=False):
    # download the tiny shakespeare dataset if not present
    input_file_path = os.path.join(os.path.dirname(__file__), filename)
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w', encoding='utf-8') as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = f.read()

    n = len(data)
    train_raw = data[:int(n*0.9)]
    val_raw = data[int(n*0.9):]

    # Define canaries and their frequencies
    # For example, we have 5 canaries each inserted a few times
    canaries = {
        "CANARY_STRING_ABC123": 2,
        "CANARY_STRING_XYZ789": 3,
        "CANARY_STRING_QWERTY": 2,
        "CANARY_STRING_12345_ABC": 1,
        "CANARY_STRING_SECRET": 2
    }

    # Convert training data to a mutable list of characters for easier insertions
    train_chars = list(train_raw)
    train_length = len(train_chars)

    if add_canaries:
        # Insert canaries at random positions in the training text
        for canary, freq in canaries.items():
            for _ in range(freq):
                # pick a random insertion point
                insert_pos = random.randint(0, train_length)
                # Insert the canary followed by a space or newline to separate from surrounding text
                canary_str = " " + canary + " "
                train_chars.insert(insert_pos, canary_str)
                # Update train_length because we've now added characters
                train_length += len(canary_str)

    # Re-join the modified training data
    train_data = "".join(train_chars)
    val_data = val_raw

    # Now encode with tiktoken gpt2 bpe
    enc = tiktoken.get_encoding("gpt2")
    train_ids = enc.encode_ordinary(train_data)
    val_ids = enc.encode_ordinary(val_data)
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # export to bin files
    train_ids = np.array(train_ids, dtype=np.uint16)
    val_ids = np.array(val_ids, dtype=np.uint16)
    train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
    val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(
        description=''
    )
    p_input  = parser.add_argument_group('Input')
    p_output = parser.add_argument_group('Output')
    p_input.add_argument(
        '-out', '--outdir',
        help="Directory for output.",
        type=str, default="", required=False
    )
    p_input.add_argument(
        '-f', '--filename',
        help="Name of file",
        type=str, default="", required=True
    )
    args = parser.parse_args()

    prep(args.filename, False)