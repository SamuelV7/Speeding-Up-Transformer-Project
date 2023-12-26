from dataclasses import dataclass
import torch
# from tqdm import tqdm
import params
import lang_tokenizer as lt
# import base_gpt as gpt
# from datasets import load_dataset
import torch.nn.functional as F
# from dataclasses import dataclass
from datetime import datetime
import json
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity, schedule, tensorboard_trace_handler

# load model 
class Models:
    code: str
    name: str
    location: str

    def __init__(self, code, model_location):
        match code:
            case "baseline":
                import base_gpt as gpt
                self.name = "baseline"
                # self.location = "models/model_4999.pt"
                self.location = model_location['baseline']
                # self.load_model(gpt)
                self.init_model(gpt)
            case "fa":
                import flash_gpt as gpt
                self.name = "fixed_attention"
                # self.location = "models/model_FA_4999.pt"
                self.location = model_location['fa']
                # self.load_model(gpt)
                self.init_model(gpt)
            case "fa_mp":
                import mixed_flash_gpt as gpt
                self.name = "fixed_attention_mixed_precision"
                # self.location = "models\model_FA_MixedPrecision_4999.pt"
                self.location = model_location['fa_mp']
                # self.load_model(gpt)
                self.init_model(gpt)
            case _:
                raise Exception("Invalid model code")
    
    def init_model(self, gpt):
        self.model = gpt.Shakespeare()
        self.model = self.model.to(params.device)

    def init_load_model(self, gpt):
        self.init_model(gpt)
        self.load_model()

    def load_model(self):
        self.model.load_state_dict(torch.load(self.location))
        self.model.eval()
        self.model.to(params.device)
    
    def get_data(self, location="tiny-shakespeare.txt"):
        with open(location, "r") as f:
            self.text = f.read()
    
    def prepare_data(self):
        self.data = torch.tensor(lt.encode(self.text), device=params.device)
        split_ix = int(len(self.data) * params.train_val_split)
        self.train_data = self.data[:split_ix]
        self.val_data = self.data[split_ix:]
    
    @torch.no_grad()
    def estimate_loss(self, cuda=False):
        out = {}
        self.model.eval()
        if cuda:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
        for split in ['train', 'val']:
            losses = torch.zeros(params.eval_interval)
            for k in range(params.eval_interval):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        if cuda:
            t2.record()
            torch.cuda.synchronize()
            out['time'] = t1.elapsed_time(t2)
        self.model.train()
        return out

    def get_batch(self, split):
        # generate a small batch of data of inputs x and targets y
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - params.block_size, (params.batch_size,))
        x = torch.stack([data[i:i+params.block_size] for i in ix])
        y = torch.stack([data[i+1:i+params.block_size+1] for i in ix])
        x, y = x.to(params.device), y.to(params.device)
        return x, y
    
    def get_and_prepare_data(self):
        self.get_data()
        self.prepare_data()

    def model_params(self):
        return sum(p.numel() for p in self.model.parameters())/1e6
    
class EvalModels:
    def __init__(self, model_location, models =["baseline", "fa", "fa_mp"], profiler=False):
        self.models = models
        self.model_location = model_location
        for model in self.models:
            self.model_eval(profiler, model)

    def model_eval(self, profiler, model):
        self.init_load_model(model)
        if profiler:
            with torch.autograd.profiler.profile(use_cuda=True) as prof:
                    # self.loss = self.model.estimate_loss(cuda=True)
                self.loss = self.loss_estimation_batched()
            self.profiler_output = prof.key_averages()
        else:
            self.loss = self.loss_estimation_test()
            # self.loss = self.model.estimate_loss()
        self.perplexity = self.estimate_perplexity()
        self.show_results()
        self.save_results()

    def load_model(self, model):
        self.model = Models(code=model, model_location=self.model_location)
        self.model.init_load_model()
        self.model.get_and_prepare_data()
    
    # def eval_training(self):
    @torch.no_grad()
    def estimate_perplexity(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(params.eval_interval)
            for k in range(params.eval_interval):
                X, Y = self.get_batch(split)
                logits, loss = self.model(X, Y)
                losses[k] = loss.item()
            out[split] = torch.exp(torch.tensor(losses.mean().clone().detach()))
        self.model.train()
        return out
    
    @torch.no_grad()
    def loss_estimation_batched(self):
        out = {}
        # losses = torch.zeros(params.eval_interval)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t1.record()
        
        X, Y = self.get_batch('val')
        logits, loss = self.model(X, Y)
        losses = loss.item()
        t2.record()
        torch.cuda.synchronize()
        out['time'] = t1.elapsed_time(t2)
        out['val'] = losses
        return out
    @torch.no_grad()
    def loss_estimation_test(self):
        out = {}
        losses = torch.zeros(params.eval_interval)
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t1.record()
        for k in range(params.eval_interval):
            X, Y = self.get_batch('val')
            logits, loss = self.model(X, Y)
            losses[k] = loss.item()
        t2.record()
        torch.cuda.synchronize()
        out['time'] = t1.elapsed_time(t2)
        out['val'] = losses.mean()
        return out


    def show_results(self):
        print("\n")
        print(f"Current model: {self.model.name}")
        print(f"Current params: {self.model.model_params()}M")
        print(f"Current loss: {self.loss}")
        print(f"Current perplexity: {self.perplexity}")
        if hasattr(self, 'profiler_output'):
            print(self.profiler_output.table(sort_by="cuda_time_total"))
        print("\n")

    def save_results(self):
        # save the accuracy as JSON
        json_data = {}
        with open(f"metrics_{self.model.name}.json", "w") as f:
            # model type
            json_data['model type'] = {
                "type": f"{self.model.name}"
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
            # inference time
            json_data['validation time '] = {
                "time": str(self.loss['time'])
            }
            # model params
            json_data['model params'] = {
                "params": f"{str(self.model.model_params())} M"
            }
            # fp32 ram usage
            json_data['fp32 ram usage'] = {
                "ram": str(torch.cuda.memory_allocated(params.device)/1e6)
            }
            json_data['loss'] = {
                # "train": str(self.loss['train']),
                "val": str(self.loss['val'])
            }
            
            json_data['perplexity test'] = str(self.perplexity['val'])

            # json.dump(json_data, f)
            json_data['date'] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            # write the JSON to file
            json.dump(json_data, f, indent=4)

            # if training:
            #     json_data['training'] = 

            print(f"Metrics saved for {self.model.name}")

class ModelTrainer:
    def __init__(self, model_code, model_location, mixed_precision, tensor_writer, profiler=True) -> None:
        self.writer = tensor_writer
        self.profiler_status = profiler
        self.model = Models(code=model_code, model_location=model_location)
        self.model.get_and_prepare_data()
        print(f"Training model {self.model.name}")
        print(f"Current params: {self.model.model_params()} M")
        
        self.the_scheduler = schedule(wait=400, warmup=5, active=20, skip_first=10)
        self.trace_handler = tensorboard_trace_handler(f'./logs/{model_code}')
        
        # self.training_metrics = 
        # time the training
        t1 = torch.cuda.Event(enable_timing=True)
        t2 = torch.cuda.Event(enable_timing=True)
        t1.record()
        # self.train(mixed_precision)
        # t2.record()
        torch.cuda.synchronize()
        # print(f"Training time: {t1.elapsed_time(t2)}")
        if self.profiler_status:
            print("Profiling...")
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, schedule=self.the_scheduler, 
                         on_trace_ready=self.trace_handler) as prof:
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, schedule=self.the_scheduler) as prof:
                self.train(mixed_precision, prof)
                t2.record()
                torch.cuda.synchronize()
                self.writer.add_scalar("Total Training Time", t1.elapsed_time(t2), 0)
                print(f"Training time: {t1.elapsed_time(t2)}")
            print(prof.key_averages().table(sort_by="cuda_time_total"))
            # print("The memory usage is: ", self.prof.key_averages().cuda_memory_usage)
            # record memory usage of cuda and cpu
            # self.writer.add_scalar("Memory Usage/CPU", self.prof.key_averages().cpu_memory_usage, 0)
            # self.writer.add_scalar("Memory Usage/CUDA", self.prof.key_averages().cuda_memory_usage, 0)
            print("Profiling done")
        else:
            self.train(mixed_precision)
            t2.record()
            torch.cuda.synchronize()
            self.writer.add_scalar("Total Training Time", t1.elapsed_time(t2), 0)
            print(f"Training time: {t1.elapsed_time(t2)}")
        # writer.close()
    def train(self, mixed_precision, the_profiler):
        if mixed_precision:
            optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=params.learning_rate)
            scaler = torch.cuda.amp.GradScaler()
    
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, schedule=self.the_scheduler) as prof:
            
            interval_t1 = torch.cuda.Event(enable_timing=True)
            interval_t1.record()
            for i in range(params.max_epochs):
                inputbatch, targetbatch = self.model.get_batch('train')
                optimizer.zero_grad()

                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    logits, loss = self.model.model(inputbatch, targetbatch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if self.profiler_status:
                    the_profiler.step()

                if i % params.eval_interval == 0 or i == params.max_epochs - 1:
                    losses = self.model.estimate_loss()
                    self.writer.add_scalar("Loss/Per Interval for Train Data", losses['train'], i)
                    self.writer.add_scalar("Loss/Per Interval for Validation Data", losses['val'], i)
                    self.writer.add_scalar("Perplexity/Per Interval for Train Data", torch.exp(losses['train']), i)
                    print(f"epoch: {i}, train loss: {losses['train']}, val loss: {losses['val']}")                    
                    torch.save(self.model.model.state_dict(), f"models_new/{self.model.name}_{i}.pt")
                    # interval time
                    interval_t2 = torch.cuda.Event(enable_timing=True)
                    interval_t2.record()
                    torch.cuda.synchronize()
                    print(f"Interval time: {interval_t1.elapsed_time(interval_t2)}")
                    self.writer.add_scalar("Time/Interval", interval_t1.elapsed_time(interval_t2), i)
                    interval_t1 = torch.cuda.Event(enable_timing=True)
                    interval_t1.record()
            # print(prof.key_averages().table(sort_by="cuda_time_total"))
        else:
            optimizer = torch.optim.AdamW(self.model.model.parameters(), lr=params.learning_rate)
            
            # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, schedule=self.the_scheduler) as prof:
            interval_t1 = torch.cuda.Event(enable_timing=True)
            interval_t1.record()
            for i in range(params.max_epochs):
                inputbatch, targetbatch = self.model.get_batch('train')
                logits, loss = self.model.model(inputbatch, targetbatch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.profiler_status:
                    the_profiler.step()

                if i % params.eval_interval == 0 or i == params.max_epochs - 1:
                    losses = self.model.estimate_loss()
                    self.writer.add_scalar("Loss/Per Interval for Train Data", losses['train'], i)
                    self.writer.add_scalar("Loss/Per Interval for Validation Data", losses['val'], i)
                    self.writer.add_scalar("Perplexity/Per Interval for Train Data", torch.exp(losses['train']), i)
                    print(f"epoch: {i}, train loss: {losses['train']}, val loss: {losses['val']}")
                    torch.save(self.model.model.state_dict(), f"models_new/{self.model.name}_{i}.pt")
                    
                    # interval time
                    interval_t2 = torch.cuda.Event(enable_timing=True)
                    interval_t2.record()
                    torch.cuda.synchronize()
                    
                    print(f"Interval time: {interval_t1.elapsed_time(interval_t2)}")
                    # writer.add_scalar("Training Time Per Interval", interval_t1.elapsed_time(interval_t2), i)
                    self.writer.add_scalar("Time/Interval", interval_t1.elapsed_time(interval_t2), i)
                    interval_t1 = torch.cuda.Event(enable_timing=True)
                    interval_t1.record()
            # print(prof.key_averages().table(sort_by="cuda_time_total"))
    

# eval_model = EvalModels(models=["fa_mp"], profiler=True)
# eval_model = EvalModels(models=["baseline", "fa", "fa_mp"], profiler=True)
models_location = {
    "baseline": "models/model_4999.pt",
    "fa": "models/model_FA_4999.pt",
    "fa_mp": "models/model_FA_MixedPrecision_4999.pt"
}
models_location2 = {
    "baseline": "models_new/baseline_4999.pt",
    "fa": "models_new/fa_4999.pt",
    "fa_mp": "models_new/fa_mp_4999.pt"
}
# about:blank#blocked

writer_baseline = SummaryWriter("runs/baseline")
writer_fa = SummaryWriter("runs/fa")
writer_fa_mp = SummaryWriter("runs/fa_mp")

use_profiler = True


# train1 = ModelTrainer(model_code="baseline", model_location=models_location2, mixed_precision=False, tensor_writer=writer_baseline, profiler=use_profiler)
train3 = ModelTrainer(model_code="fa", model_location=models_location2, mixed_precision=False, tensor_writer=writer_fa, profiler=use_profiler)
train2 = ModelTrainer(model_code="fa_mp", model_location=models_location2, mixed_precision=True, tensor_writer=writer_fa_mp, profiler=use_profiler)