import lang_tokenizer as lt
import torch
def set_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        return device
        # torch.set_default_tensor_type(torch.cuda.FloatTensor)
        # check if MPS is available
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        return device
        # torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        device = torch.device('cpu')
        return device

# hyperparams of the network
batch_size = 32
block_size = 314 # this is max_len
dropout = 0.3
d_model = 128 # also called n_embd
nhead = 4
num_decoder_layers = 3
learning_rate = 1e-4
max_epochs = 5000
eval_interval = 10
vocab_size = lt.the_vocab_size
device = set_device()