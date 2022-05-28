import json
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-file", type=str)
    parser.add_argument("--inference-result", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-replace-num", type=int, default=4)
    
    return parser.parse_args()

args = get_args()

refs = []
infer_res = []

fref = open(args.reference_file, "r", encoding="utf-8")
refs = [json.loads(line) for _, line in tqdm(enumerate(fref))]
fref.close()

finf = open(args.inference_result, "r", encoding="utf-8")
infer_res = finf.readlines()
finf.close()

output = []
for _, (ref, infer_line) in tqdm(enumerate(zip(refs, infer_res))):
    output.append({
        "summary": ref["summary"],
        "text": ref["text"],
        "polluted-summary": infer_line
    })

with open(args.output_file, "w", encoding="utf-8") as fout:
    for item in output:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
