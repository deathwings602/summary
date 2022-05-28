import json

SOURCE_FILE = "/home/huangshuhong/data/dataprocess/data/guangMingDaily.json"
TARGET_FILE = "/home/huangshuhong/data/dataprocess/data/guangMingDaily-2.json"

json_file = json.load(open(SOURCE_FILE, "r", encoding="utf-8"))

for news in json_file:
    content = news["content"]
    if (len(content) > 1 and ('\u4e00' <= content[0][-1] <= '\u9fa5')):
        news["content"] = [content[0] + content[1]] + content[2:]
        

json_lines = [json.dumps(item, ensure_ascii=False) + "\n" for item in json_file]

with open(TARGET_FILE, "w", encoding="utf-8") as f:
    f.writelines(json_lines)
