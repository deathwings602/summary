import json

SOURCE_FILE = "/home/huangshuhong/data/dataprocess/infer_results/guangMingDaily-2.json.1800/3-0.jsonl.0"
TARGET_FILE = "/home/huangshuhong/data/dataprocess/data/EcoNewsSum_processed/guangMingDaily-2-se.jsonl"

json_lines = open(SOURCE_FILE, "r", encoding="utf-8").readlines()

with open(TARGET_FILE, "w", encoding="utf-8") as f:
    for line in json_lines:
        news_info = json.loads(line)
        permuted_news_info = {
            "id": news_info["id"],
            "shannon_entropy": news_info["shannon_entropy"],
            "summary": news_info["summary"],
            "text": news_info["text"]
        }
        f.write(json.dumps(permuted_news_info, ensure_ascii=False) + "\n")
