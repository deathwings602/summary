from utils import *
import numpy as np

args = get_args()

fin = open(args.input_file, "r", encoding="utf-8")
fout = open(args.output_file, "w", encoding="utf-8")

input_lines = fin.readlines()
swap_nums = []

for line in tqdm(input_lines, total=len(input_lines)):
    item = json.loads(line)
    polluted_summ, swap_num = swap_ner(item["summary"], item["summ-entity"], item["text-entity"], args.max_replace_num, args.min_length)
    to_write = {
        "summary": item["summary"],
        "text": item["text"],
        "polluted-summary": polluted_summ
    }
    swap_nums.append(swap_num)
    if swap_num > 0:
        fout.write(json.dumps(to_write, ensure_ascii=False) + "\n")

fin.close()
fout.close()

swap_nums = np.array(swap_nums)
print(f"{np.sum(swap_nums > 0)}  {np.average(swap_nums)}")
