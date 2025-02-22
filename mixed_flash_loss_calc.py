import torch
import flash_gpt as gpt
import params
import lang_tokenizer as lt
import json

model_name = "models\model_FA_MixedPrecision_4999.pt"
# load model for accuracy test
model = gpt.Shakespeare()
model.load_state_dict(torch.load(model_name, map_location=params.device))
model.eval()
model.to(params.device)

with open("tiny-shakespeare.txt", "r") as f:
    text = f.read()

data = torch.tensor(lt.encode(text), device=params.device)
split_ix = int(len(data) * params.train_val_split)
train_data = data[:split_ix]
val_data = data[split_ix:]

# start context
# context = torch.zeros((1,1), dtype=torch.long, device=params.device)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - params.block_size, (params.batch_size,))
    x = torch.stack([data[i:i+params.block_size] for i in ix])
    y = torch.stack([data[i+1:i+params.block_size+1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y

@torch.no_grad()
def loss_estimation():
    out = {}
    losses = torch.zeros(params.eval_interval)
    t1 = torch.cuda.Event(enable_timing=True)
    t2 = torch.cuda.Event(enable_timing=True)
    t1.record()
    for k in range(params.eval_interval):
        X, Y = get_batch('val')
        logits, loss = model(X, Y)
        losses[k] = loss.item()
    t2.record()
    torch.cuda.synchronize()
    out['time'] = t1.elapsed_time(t2)
    out['val'] = losses.mean()
    return out

with torch.autograd.profiler.profile(use_cuda=True) as prof:
    loss = loss_estimation()
profiler_output = prof.key_averages()
print(profiler_output.table(sort_by="cuda_time_total"))

# save the accuracy as JSON
loss = loss_estimation()
json_data = {}
with open("mixed_attention.json", "w") as f:
    # model type
    json_data['model type'] = {
        "type": "flash-attention Mixed Precision"
    }
    # add current params to the JSON
    json_data['h-params'] = {
        "batch_size": params.batch_size,
        "block_size": params.block_size,
        "dropout": params.dropout,
        "n_embeddings": params.n_embeddings,
        "nhead": params.nhead,
        "num_decoder_layers": params.num_decoder_layers,
        "learning_rate": params.learning_rate,
        "max_epochs": params.max_epochs,
        "eval_interval": params.eval_interval,
        "vocab_size": params.vocab_size,
        "train_val_split": params.train_val_split,
        "device": str(params.device)
    }
    #inference time
    json_data['inference time'] = {
        "time": str(loss['time'])
    }
    # model params
    json_data['model params'] = {
        "params": str(sum(p.numel() for p in model.parameters()))
    }
    # fp32 ram usage
    json_data['fp32 ram usage'] = {
        "ram": str(torch.cuda.memory_allocated(params.device)/1e6)
    }
    json_data['loss test'] = {
        "val": str(loss['val'])
    }
    json.dump(json_data, f)