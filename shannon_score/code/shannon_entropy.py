import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import json
from model_center import model
from model_center.model import CPM1Config,CPM1 
from model_center.tokenizer import CPM1Tokenizer 
from tqdm import tqdm

import torch.distributed as dist
from model_center import get_args
from cpm1_dataset import DATASET, collate_fn
from model_center.dataset import DistributedDataLoader

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer(args.vocab_file)
    return tokenizer

def get_model(args, vocab_size):
    """
    config = CPM1Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))

    model = CPM1(config).cuda()
    # if args.load != None:
    model.load_state_dict(
        torch.load(args.load),
        strict = True
    )
    torch.cuda.synchronize()"""
    model = CPM1.from_pretrained(args.load).cuda()
    return model

def setup_model(args):
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer.vocab_size)
    return tokenizer, model

def initialize():
    # get arguments
    args = get_args()
    # init bmp 
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group("nccl")

    # init save folder
    if args.save != None:
        os.makedirs(args.save, exist_ok=True)
    return args

def prepare_dataset(args, tokenizer):
    dataset = DATASET[args.dataset_name](args.input_file, tokenizer, args.max_length)
    my_collate_fn = lambda batch_data: collate_fn(batch_data, args.max_length, tokenizer)
    dataloader = DistributedDataLoader(dataset, shuffle=False, collate_fn=my_collate_fn, batch_size=args.batch_size)

    return dataloader

@torch.no_grad()
def main():
    args = initialize()
    tokenizer, model = setup_model(args)

    fout = open("{}.{}".format(args.output_file, args.local_rank), "w", encoding="utf-8")
    dataloader = prepare_dataset(args, tokenizer)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=1000000)
    
    def cal_entropy(input_dict):
        input_tokens = input_dict["input_tokens"].cuda()
        length = input_dict["input_length"].cuda()
        context = input_dict["context"].cuda()
        input_span = input_dict["input_span"].cuda()
        target = input_dict["target"].cuda()
        target_length = input_dict["target_length"]
        output,_ = model(input_tokens, length, context, input_span)
        losses = torch.zeros(len(input_tokens))
        
        for i in range(len(losses)):
            ot = output[i] / 100
            # print(ot[: 10])
            tar = target[i]
            #l = length[i].item()
            #tl = target_length[i].item()
            #ot = ot[l - tl - 1: l - 1]
            #tar = tar[l - tl - 1: l - 1]
            #print(ot.size(), tar.size(), ot.dtype, tar.dtype)
            #print(tar)
            losses[i] = criterion(ot, tar).item()
        # print(losses)
        return losses

    for input_dict in dataloader:
        SD_losses = cal_entropy(input_dict["S-D"])
        DD_losses = cal_entropy(input_dict["D-D"])
        D_losses = cal_entropy(input_dict["D"])
        # print(f"{SD_losses}, {DD_losses}, {D_losses}")
        shannon_entropy = (D_losses - SD_losses) / (D_losses - DD_losses)
        decoded = input_dict["decoded"]
        for i in range(len(decoded)):
            decoded[i]["shannon_entropy"] = shannon_entropy[i].item()
        for item in decoded:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    fout.close()

if __name__ == "__main__":
    main()
