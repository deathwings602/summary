import random
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--sample-num", type=int, default=100)

    return parser.parse_args()

args = get_args()
input_data = []
with open(args.input_file, "r", encoding="utf-8") as fin:
    for _, line in tqdm(enumerate(fin)):
        input_data.append(line)

output_data = random.sample(input_data, args.sample_num)
with open(args.output_file, "w", encoding="utf-8") as fout:
    for line in tqdm(output_data):
        fout.write(line)