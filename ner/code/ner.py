from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, BertTokenizerFast, pipeline
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--batch-size", type=int, default=4)
    
    return parser.parse_args()

def process_bad_int64(input_dicts: list):
    output_dicts = []
    for input_dict in input_dicts:
        output_dicts.append({
            "word": input_dict["word"],
            "entity": input_dict["entity"],
            "index": int(input_dict["index"]),
            "start": int(input_dict["start"]),
            "end": int(input_dict["end"])
        })
    return output_dicts

class LCSTS_Dataset(Dataset):
    def __init__(self, path: str) -> None:
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for _, line in enumerate(f):
                input_dict = json.loads(line)
                self.data.append({
                    "summary": input_dict["summary"],
                    "text": input_dict["text"]
                })
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index: int):
        return self.data[index]


tokenizer = BertTokenizerFast.from_pretrained('bert-base-chinese')

model = AutoModelForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ner")

pipe = pipeline('ner', model=model, tokenizer=tokenizer, device=0)

args = get_args()

dataset = LCSTS_Dataset(args.input_file)
summ_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda input: [item["summary"] for item in input])
text_dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=lambda input: [item["text"] for item in input])

fout = open(args.output_file, "w", encoding="utf-8")

for summ, text in tqdm(zip(summ_dataloader, text_dataloader), total=len(dataset) // args.batch_size):
    summ_entity = pipe(summ)
    text_entity = pipe(text)
    
    for s, t, se, te in zip(summ, text, summ_entity, text_entity):
        se = process_bad_int64(se)
        te = process_bad_int64(te)
        fout.write(json.dumps({"summary": s, "text": t, "summ-entity": se, "text-entity": te}, ensure_ascii=False) + "\n")
    fout.flush()

fout.close()
