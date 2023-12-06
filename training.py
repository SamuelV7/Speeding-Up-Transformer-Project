from datetime import time
import gpt
import params
import torch
import lang_tokenizer as lt

# read text data 
with open("tiny-shakespeare.txt", "r") as f:
    text = f.read()

model = gpt.Shakespeare()
model = model.to(params.device)
data = torch.tensor(lt.encode(text), device=params.device)
split_ix = int(len(data) * params.train_val_split)
train_data = data[:split_ix]
val_data = data[split_ix:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - params.block_size, (params.batch_size,))
    x = torch.stack([data[i:i+params.block_size] for i in ix])
    y = torch.stack([data[i+1:i+params.block_size+1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(params.eval_interval)
        for k in range(params.eval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# calculate params
print(sum(p.numel() for p in model.parameters())/1e6, "M")

optimizer = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
scaler = torch.cuda.amp.GradScaler()

for i in range(params.max_epochs):
    inputbatch, targetbatch = get_batch('train')
    # logits, loss = model(inputbatch, targetbatch)
    optimizer.zero_grad()
    
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        logits, loss = model(inputbatch, targetbatch)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    # loss.backward()
    # optimizer.step()
    if i % params.eval_interval == 0 or i == params.max_epochs - 1:
        losses = estimate_loss()
        print(f"epoch: {i}, train loss: {losses['train']}, val loss: {losses['val']}")
        torch.save(model.state_dict(), f"model_flash_attention_ma{i}.pt")