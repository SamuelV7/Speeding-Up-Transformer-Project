import gpt
import params
import torch
import lang_tokenizer as lt


# load the model - classic model without mixed precision and flash attention
model = gpt.Shakespeare()
model.load_state_dict(torch.load("model_4999.pt"))
model.eval()
model.to(params.device)

# # generate text
context = torch.zeros((1,1), dtype=torch.long, device=params.device)
# output = model.generate(context, 500)[0].tolist()
# print("the output", lt.decode(output))
while True:
    output = model.generate(context, 3000)[0].tolist()
    print(lt.decode(output))

    # press enter to continue
    input()