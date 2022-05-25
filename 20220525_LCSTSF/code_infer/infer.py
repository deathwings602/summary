#coding:utf-8

import torch
import numpy as np
import os
from model_center.model import CPM1Config,CPM1 
from model_center.tokenizer import CPM1Tokenizer 
from tqdm import tqdm
import torch.distributed as dist

from model_center import get_args
from generation import generate

from infer_dataset import INFER_DATASET, BatchInferDataset

def get_tokenizer(args):
    tokenizer = CPM1Tokenizer(args.vocab_file)
    return tokenizer

def get_model(args, vocab_size):
    config = CPM1Config.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    print ("vocab size:%d"%(vocab_size))

    model = CPM1(config).cuda()
    # if args.load != None:
    model.load_state_dict(
        torch.load(args.load),
        strict = True
    )
    torch.cuda.synchronize()
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


def main():
    args = initialize()
    tokenizer, model = setup_model(args)

    fout = open("{}.{}".format(args.output_file, args.local_rank), "w", encoding="utf-8")

    dataset = INFER_DATASET[args.dataset_name](args.input_file, args.max_length)
    total_lines = dataset.total_length
    step = (total_lines + dist.get_world_size() -1) // dist.get_world_size()
    dataset.read_dataset(step * args.local_rank, step * (args.local_rank + 1), tokenizer)
    batch_num = (step + args.batch_size - 1) // args.batch_size
    batch_dataset = BatchInferDataset(dataset, tokenizer, args.span_length, args.batch_size, batch_num)
    min_len = 2 # 确保生成内容不为空
    def work(input_dict):
        result = generate(model, tokenizer, input_dict, beam=args.beam_size,
                        temperature = args.temperature, length_penalty=args.length_penalty, top_k = args.top_k, top_p = args.top_p,
                        no_repeat_ngram_size = args.no_repeat_ngram_size, repetition_penalty = args.repetition_penalty, 
                        random_sample=args.random_sample, min_len=min_len)
        for idx, sent in enumerate(result):
            fout.write(sent + '\t' + str(input_dict['ids'][idx]) + '\n')
            fout.flush()

    if args.local_rank == 0:
        for input_dict in tqdm(batch_dataset):
            if input_dict['valid']:
                work(input_dict)
    else:
        for input_dict in batch_dataset:
            if input_dict['valid']:
                work(input_dict)
        
    fout.close()

if __name__ == "__main__":
    main()
