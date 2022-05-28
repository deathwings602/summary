import torch
import json

import numpy as np
import bmtrain as bmt


class CNewSum_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, split, rank, world_size, tokenizer, max_length) -> None:
        self.data = []
        path = f"{path}/CNewSum/{split}.jsonl.{max_length}"
        bmt.print_rank(f"Start loading dataset {path}")
        if split == 'test':
            pass
        else:
            with open(path, encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    if i % 5000 == 0:
                        bmt.print_rank(i)
                    line_json = json.loads(line)
                    lef_tokens = line_json['lef_tokens']
                    rig_tokens = line_json['rig_tokens']

                    input_tokens, input_length, context, input_span, target, target_length = self.make_input(lef_tokens, rig_tokens, max_length, tokenizer)

                    self.data.append({
                        "input_tokens": input_tokens,
                        "input_length": input_length,
                        "input_context": context,
                        "input_span": input_span,
                        "targets": target,
                        "target_length": target_length,
                    })

    def make_input(self, lef_tokens, rig_tokens, max_length, tokenizer):
        lef_length = len(lef_tokens)
        rig_length = len(rig_tokens)

        input = lef_tokens + rig_tokens

        length = len(input)

        assert length < max_length

        input_tokens = torch.zeros((max_length,), dtype=torch.int32)
        input_tokens[:length] = torch.tensor(input).int()

        input_length = torch.tensor(length, dtype=torch.int32)
        target_length = torch.tensor(rig_length, dtype=torch.int32)
        
        context = np.arange(max_length)
        context = (context < lef_length) | (context >= length)
        context = torch.from_numpy(context).bool()

        target = np.full((max_length,), -100)
        target[lef_length-1:length-1] = rig_tokens
        target = torch.from_numpy(target).int()

        input_span = torch.zeros((max_length,), dtype=torch.int32)

        return input_tokens, input_length, context, input_span, target, target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class LCSTS_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_length) -> None:
        self.data = []
        # path = f"{path}/EcoNewSum/{split}.json"
        # bmt.print_rank(f"Start loading dataset {path}")
        if False:
            pass
        else:
            with open(path, encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    #if i % 5000 == 0:
                    #    bmt.print_rank(i)
                    line_json = json.loads(line)
                    lef_tokens = line_json['lef_tokens'][1:]
                    rig_tokens = line_json['rig_tokens'][: -1]
                    self.data.append({
                        # "id": line_json["id"],
                        "lef_tokens": lef_tokens,
                        "rig_tokens": rig_tokens
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class EcoNewSum_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_length) -> None:
        self.data = []
        # path = f"{path}/EcoNewSum/{split}.json"
        # bmt.print_rank(f"Start loading dataset {path}")
        if False:
            pass
        else:
            with open(path, encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    #if i % 5000 == 0:
                    #    bmt.print_rank(i)
                    line_json = json.loads(line)
                    summary = line_json['content'][0].replace(' ', '')
                    text = ''.join(line_json['content'][1:]).replace(' ', '')
                    self.data.append({
                        "summary": summary,
                        "text": text
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class AllSum_Dataset(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, max_length) -> None:
        self.data = []
        # path = f"{path}/EcoNewSum/{split}.json"
        # bmt.print_rank(f"Start loading dataset {path}")
        if False:
            pass
        else:
            with open(path, encoding='utf8') as fin:
                for i, line in enumerate(fin):
                    #if i % 5000 == 0:
                    #    bmt.print_rank(i)
                    line_json = json.loads(line)
                    self.data.append({
                        "summary": line_json["summary"],
                        "text": line_json["text"]
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

DATASET = {
    "LCSTS": AllSum_Dataset,
    "CNewSum": CNewSum_Dataset,
    "EcoNewSum": EcoNewSum_Dataset,
    "MTGOOD": AllSum_Dataset
}

def make_input(lef_tokens, rig_tokens, max_length, tokenizer):
    half_length = (max_length - 1) // 2
    lef_tokens = lef_tokens[: half_length]
    rig_tokens = rig_tokens[: half_length] + [tokenizer.eod_id]
    lef_length = len(lef_tokens)
    rig_length = len(rig_tokens)

    input = lef_tokens + rig_tokens

    length = len(input)
    assert length < max_length

    input_tokens = torch.zeros((max_length,), dtype=torch.int32)
    input_tokens[:length] = torch.tensor(input).int()

    input_length = torch.tensor(length, dtype=torch.int32)
    target_length = torch.tensor(rig_length, dtype=torch.int32)
    
    context = np.arange(max_length)
    context = (context < lef_length) | (context >= length)
    context = torch.from_numpy(context).bool()

    target = np.full((max_length,), 1000000)
    target[lef_length-1:length-1] = rig_tokens
    target = torch.from_numpy(target).long()

    input_span = torch.zeros((max_length,), dtype=torch.int32)

    return input_tokens, input_length, context, input_span, target, target_length


def make_input_strategy(batch_data, max_length, tokenizer, strategy: str):
    input_tokens, input_length, context, input_span, target, target_length = [], [], [], [], [], []
    for item in batch_data:
        lef_tokens = item["lef_tokens"]
        rig_tokens = item["rig_tokens"]
        if strategy == "S-D":
            it, il, ctx, isn, tar, tl = make_input([1] + lef_tokens, rig_tokens, max_length, tokenizer)
        elif strategy == "D-D":
            it, il, ctx, isn, tar, tl = make_input([1] + rig_tokens, rig_tokens, max_length, tokenizer)
        elif strategy == "D":
            it, il, ctx, isn, tar, tl = make_input([1], rig_tokens, max_length, tokenizer)
        else:
            raise RuntimeError(f"Unkown strategy {strategy} for function make_input_strategy")
        input_tokens.append(it)
        input_length.append(il)
        context.append(ctx)
        input_span.append(isn)
        target.append(tar)
        target_length.append(tl)

    input_tokens = torch.stack(input_tokens)
    input_length = torch.stack(input_length)
    context = torch.stack(context)
    input_span = torch.stack(input_span)
    target = torch.stack(target)
    target_length = torch.stack(target_length)
    input_dict = {
        "input_tokens": input_tokens,
        "input_length": input_length,
        "context": context,
        "input_span": input_span,
        "target": target,
        "target_length": target_length
    }
    return input_dict


def tokenize(summary, text, tokenizer):
    #lef_tokens = [1] + tokenizer.encode('“') + tokenizer.encode(text)[:max_length] + tokenizer.encode('”的摘要是:')
    #rig_tokens = tokenizer.encode(summary) + [tokenizer.eod_id]
    lef_tokens = tokenizer.encode(summary)
    rig_tokens = tokenizer.encode(text)

    return lef_tokens, rig_tokens


def encode_all(batch_data, tokenizer):
    result = []
    for data in batch_data:
        lef, rig = tokenize(data["summary"], data["text"], tokenizer)
        result.append({
            "lef_tokens": lef,
            "rig_tokens": rig
        })
    return result


def decode_all(batch_data, tokenizer):
    decoded = []
    for item in batch_data:
        summary = tokenizer.decode(item["lef_tokens"])
        text = tokenizer.decode(item["rig_tokens"])
        decoded.append({
            # "id": item["id"],
            "summary": summary,
            "text": text
        })
    return decoded


def collate_fn(batch_data, max_length, tokenizer):
    # S-D
    tokenized_data = encode_all(batch_data, tokenizer)
    input_dict = {
        "S-D": make_input_strategy(tokenized_data, max_length, tokenizer, "S-D"),
        "D-D": make_input_strategy(tokenized_data, max_length, tokenizer, "D-D"),
        "D": make_input_strategy(tokenized_data, max_length, tokenizer, "D"),
        "decoded": batch_data
    }
    
    return input_dict

