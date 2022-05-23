from utils import *

args = get_args()

fin = open(args.input_file, "r", encoding="utf-8")
fout = open(args.output_file, "w", encoding="utf-8")

input_lines = fin.readlines()

for line in tqdm(input_lines, total=len(input_lines)):
    item = json.loads(line)
    summ_entities = get_entity_list(item["summ-entity"])
    text_entities = get_entity_list(item["text-entity"])
    to_write = {
        "summary": item["summary"],
        "text": item["text"],
        "summ-entity": summ_entities,
        "text-entity": text_entities
    }
    fout.write(json.dumps(to_write, ensure_ascii=False) + "\n")

fout.close()
fin.close()

