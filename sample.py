"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
import torch.nn.functional as F

# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out'  # ignored if init_from is not 'resume'
start = "\n"  # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10  # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8  # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200  # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
compile = False  # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model)  # requires PyTorch 2.0 (optional)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:  # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]


# Sampling and entropy computation
def sample_with_entropy(model, x, max_new_tokens, temperature, top_k):
    generated = x
    entropies = []

    # Compute entropy for the initial tokens
    logits, _ = model(generated)
    probabilities = F.softmax(logits, dim=-1)
    token_entropies = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=-1)
    entropies.extend(token_entropies[0].tolist())  # Add entropies for the initial prompt

    for _ in range(max_new_tokens):
        # Get logits for the last token
        logits, _ = model(generated)
        logits = logits[:, -1, :]  # Only the last token's logits
        probabilities = F.softmax(logits / temperature, dim=-1)

        # Calculate entropy for the last token
        token_entropy = -(probabilities * torch.log(probabilities + 1e-9)).sum(dim=-1)
        entropies.append(token_entropy.item())

        # Sample the next token
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        # Append the sampled token
        generated = torch.cat((generated, next_token), dim=1)

    return generated, entropies

# Generate samples
with torch.no_grad():
    with ctx:
        matches = 0
        for i in range(num_samples):
            generated, entropies = sample_with_entropy(model, x, max_new_tokens, temperature, top_k)
            tokens = generated[0].tolist()
            print(f"Sample {i + 1}:")
            print("Generated Text and Per-token Entropies:")
            found = False
            canaries = {
                "CANARY_STRING_ABC123": 2,
                "CANARY_STRING_XYZ789": 3,
                "CANARY_STRING_QWERTY": 2,
                "CANARY_STRING_12345_ABC": 1,
                "CANARY_STRING_SECRET": 2
            }

            for idx, token_id in enumerate(tokens):
                token = decode([token_id])
                if token in canaries:
                    matches += 1
                entropy = entropies[idx] if idx < len(entropies) else "N/A"
                print(f"Token: {token} | Entropy: {entropy:.4f}" if isinstance(entropy, float) else f"Token: {token} | Entropy: {entropy}")
            print("Generated Text:")
            print(decode(tokens))
            print('-' * 40)
        print(f"NUMBER OF MATCHES: {matches}")

def entropy_to_color(entropy, min_ent=0.0, max_ent=1.0):
    # Clamp entropy to range
    e = max(min_ent, min(entropy, max_ent))
    # Calculate a ratio
    ratio = (e - min_ent) / (max_ent - min_ent)
    # ratio=0 -> green (#00FF00), ratio=1 -> red (#FF0000)
    # We can interpolate in RGB space:
    # green = (0, 255, 0), red = (255, 0, 0)
    r = int(ratio * 255)
    g = int((1 - ratio) * 255)
    b = 0
    return f"rgb({r},{g},{b})"

html_output = []
html_output.append("<html><body><p style='font-size:16px;'>")

for token, ent in zip(tokens, entropies):
    color = entropy_to_color(ent, min_ent=0.0, max_ent=1.0)
    # Escape HTML special chars in token if needed
    safe_token = token.replace("<", "&lt;").replace(">", "&gt;")
    html_output.append(f"<span style='color:{color};'>{safe_token}</span>")

html_output.append("</p></body></html>")
html_str = "".join(html_output)

# # Write to a file
# with open("entropy_colored_output.html", "w", encoding="utf-8") as f:
#     f.write(html_str)
