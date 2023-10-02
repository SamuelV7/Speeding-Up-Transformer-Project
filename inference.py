import gpt
import params
import torch
import lang_tokenizer as lt


# load the model
model = gpt.Shakespeare()
model.load_state_dict(torch.load("model_4999.pt"))

# generate text
context = torch.zeros((1,1), dtype=torch.long, device=params.device)

while True:
    output = model.generate(context, max_output=500)[0].tolist()
    print(lt.decode(output))

    # press enter to continue
    input()