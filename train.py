import os
import time
import math
import pickle
from contextlib import nullcontext
import inspect
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT
from sparsity import *  # Assuming you need this

# -----------------------------------------------------------------------------
# Default configuration
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
wandb_log = False
wandb_project = 'owt'
wandb_run_name = 'gpt2'
dataset = 'openwebtext'
gradient_accumulation_steps = 1
batch_size = 12
block_size = 1024
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5
backend = 'nccl'
device = 'cuda'
dtype = 'float32'  # We'll finalize to float32 even if overridden
compile = False
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # may override some settings
config = {k: globals()[k] for k in config_keys}

# Force dtype to float32 for DP stability
dtype = 'float32'

ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
else:
    master_process = True
    seed_offset = 0
    ddp_world_size = 1

tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ctx = nullcontext()  # no autocast

data_dir = os.path.join('data', dataset)

class UniformBatchDataset(torch.utils.data.Dataset):
    def __init__(self, data_file, block_size, batch_size):
        self.block_size = block_size
        self.batch_size = batch_size
        data = np.fromfile(data_file, dtype=np.uint16)
        data = torch.from_numpy(data.astype(np.int64))

        # Trim data so that (len(data)-1)//block_size is divisible by batch_size
        length_in_seq = (len(data) - 1) // block_size
        # number of full batches
        full_batches = (length_in_seq // batch_size) * batch_size
        final_length = full_batches * block_size + 1
        self.data = data[:final_length]

    def __len__(self):
        return (len(self.data) - 1) // self.block_size

    def __getitem__(self, idx):
        i = idx * self.block_size
        x = self.data[i:i+self.block_size]
        y = self.data[i+1:i+1+self.block_size]
        return x, y

train_data_file = os.path.join(data_dir, 'train.bin')
val_data_file = os.path.join(data_dir, 'val.bin')

train_dataset = UniformBatchDataset(train_data_file, block_size, batch_size)
val_dataset = UniformBatchDataset(val_data_file, block_size, batch_size)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    pin_memory=(device_type=='cuda')
)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=(device_type=='cuda')
)

meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    if master_process:
        print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

iter_num = 0
best_val_loss = 1e9

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout)
if init_from == 'scratch':
    if master_process:
        print("Initializing a new model from scratch")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    if master_process:
        print(f"Resuming training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    if master_process:
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)

sparsity_level = 50
sparsify_threshold_based_global(model, sparsity_level)

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device, dtype=torch.float32)

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
# Force no fused if it causes issues:
for param_group in optimizer.param_groups:
    param_group.pop('fused', None)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

# Do not use torch.compile() for now
# if compile:
#     if master_process:
#         print("compiling the model...")
#     model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[int(device.split(':')[-1])])

from opacus import PrivacyEngine

noise_multiplier = 1.0
max_grad_norm = 1.0

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=max_grad_norm,
    # secure_mode=False by default
)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    def eval_split(loader):
        losses = []
        count = 0
        for X, Y in loader:
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses.append(loss.item())
            count += 1
            if count >= eval_iters:
                break
        return float(np.mean(losses))

    out['train'] = eval_split(train_loader)
    out['val'] = eval_split(val_loader)
    model.train()
    return out

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

t0 = time.time()
local_iter_num = 0
raw_model = model.module if ddp else model
running_mfu = -1.0

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

while True:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100,
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                if master_process:
                    print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

    if iter_num == 0 and eval_only:
        break

    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, loss = model(X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item()
            if local_iter_num >= 5:
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")

        iter_num += 1
        local_iter_num += 1

        if master_process and iter_num % eval_interval == 0:
            epsilon = privacy_engine.get_epsilon(delta=1e-5)
            print(f"After {iter_num} steps, DP ε = {epsilon:.2f}")

        if iter_num > max_iters:
            break
    if iter_num > max_iters:
        break

if ddp:
    destroy_process_group()
