import torch
from tqdm import tqdm
import params
import lang_tokenizer as lt
import base_gpt as gpt
from datasets import load_dataset
import torch.nn.functional as F

max_length = params.block_size
# stride = 512
stride = 128
seq_len = params.vocab_size
# params.device = 'cpu'

model = gpt.Shakespeare() 
model.load_state_dict(torch.load("models\model_4999.pt"))
model.eval()
model.to(params.device)

#load wiki datasetdid ya see this 
# dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test')
# vocab_size, encoder, decoder = lt.tokenizer("\n\n".join(dataset['text']))
# encodings = encoder("\n\n".join(dataset['text']))
# encodings = torch.tensor(encodings).to(params.device)

with open("tiny-shakespeare.txt", "r") as f:
    text = f.read()

# model = gpt.Shakespeare()
# model = model.to(params.device)
data = torch.tensor(lt.encode(text), device=params.device)
split_ix = int(len(data) * params.train_val_split)
train_data = data[:split_ix]
val_data = data[split_ix:]


def preplexity_hf(max_length, stride, seq_len, model, encodings):
    nlls = []
    prev_end_loc = 0
    context = torch.zeros((1,1), dtype=torch.long, device=params.device)

    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len, len(encodings))
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    # input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        input_ids = torch.tensor(encodings[begin_loc:end_loc]).to(params.device)
    # make it 2 dimensional
        input_ids = input_ids.unsqueeze(0)
    # remove the last token 
        input_ids = input_ids[:, :-1]
    # input_ids = lt.encode(tmp_list).to(params.device)
        target_ids = input_ids.clone()
        target_ids.to(params.device)
        target_ids[:, -trg_len+1] = -100

        with torch.no_grad():
            outputs = model.forward(input_ids, targets=target_ids)
        # set context to the last token of the past
            context = outputs[0][:, -1, :].unsqueeze(1)
        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
            idx, neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    return nlls

# nlls = preplexity_hf(max_length, stride, seq_len, model, encodings)

# ppl = torch.exp(torch.stack(nlls).mean())
# print(ppl)

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - params.block_size, (params.batch_size,))
    x = torch.stack([data[i:i+params.block_size] for i in ix])
    y = torch.stack([data[i+1:i+params.block_size+1] for i in ix])
    x, y = x.to(params.device), y.to(params.device)
    return x, y

def estimate_perplexity():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(params.eval_interval)
        for k in range(params.eval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        # out[split] = torch.exp(torch.tensor(losses.mean()))
        out[split] = torch.exp(torch.tensor(losses.mean().clone().detach()))
    model.train()
    return out

score = estimate_perplexity()
print(score)