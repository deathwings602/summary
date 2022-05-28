import json
import tqdm
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--percentage", type=float, default=0.7)
    
    return parser.parse_args()

args = get_args()
json_lines = [json.loads(line) for line in open(args.input_file, "r", encoding="utf-8")]

json_lines.sort(key=lambda x: -x["shannon_entropy"])
reserved_num = int(len(json_lines) * args.percentage)

with open(args.output_file, "w", encoding="utf-8") as fout:
    for item in json_lines[: reserved_num]:
        fout.write(json.dumps(item, ensure_ascii=False) + "\n")
