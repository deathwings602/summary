import json
import math

THRESHOLD = 0.15

MIN_LENGTH = 100

SOURCE_FILE = "/home/huangshuhong/data/dataprocess/infer_results/economicInfoDaily-2.json/3-0.jsonl.0"
TARGET_FILE = "/home/huangshuhong/data/dataprocess/data/EcoNewSum_filtered/economicInfoDaily-2-filtered.jsonl"

def good_summary(news_info: dict):
    if news_info["shannon_entropy"] < THRESHOLD or \
            news_info["shannon_entropy"] == float("inf") or \
            math.isnan(news_info["shannon_entropy"]) or \
            len(news_info["text"]) < MIN_LENGTH:
        return False
    return True

json_lines = open(SOURCE_FILE, "r", encoding="utf-8").readlines()

with open(TARGET_FILE, "w", encoding="utf-8") as f:
    for line in json_lines:
        news_info = json.loads(line)
        if good_summary(news_info):
            f.write(json.dumps(news_info, ensure_ascii=False) + "\n")
