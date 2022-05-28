import json
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    
    return parser.parse_args()


def process(line_json: dict):
    summ_list = line_json["s"]
    text_list = line_json["t"]
    summary = "。".join(summ_list) + "。"
    text = "。".join(text_list) + "。"
    result = {
        "id": line_json["id"],
        "summary": summary,
        "text": text
    }
    
    return result


def get_statistics(line_jsons: list):
    summ_len = np.array([len(line_json["summary"]) for line_json in line_jsons])
    text_len = np.array([len(line_json["text"]) for line_json in line_jsons])
    
    summ_len = np.sort(summ_len)
    text_len = np.sort(text_len)
    
    threshold = 0.98
    cut = int(threshold * len(summ_len))
    print(f"summary mean: {np.mean(summ_len)}, max: {np.max(summ_len)}, cut: {summ_len[cut]}")
    print(f"text mean: {np.mean(text_len)}, max: {np.max(text_len)}, cut: {text_len[cut]}")


args = get_args()

input_json = [json.loads(line) for line in open(args.input_file, "r", encoding="utf-8")]
output_json = [process(line) for line in input_json]

get_statistics(output_json)

with open(args.output_file, "w", encoding="utf-8") as fout:
    for line in output_json:
        fout.write(json.dumps(line, ensure_ascii=False) + "\n")
