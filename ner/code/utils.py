import random
import argparse
import json
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--min-length", type=int, default=2)
    parser.add_argument("--max-replace-num", type=int, default=4)
    
    return parser.parse_args()


def load_data(input_file: str):
    f = open(input_file, "r", encoding="utf-8")
    res = [json.loads(line) for line in tqdm(f, total=10)]
    f.close()
    return res


def save_data(output_file: str, data: list):
    with open(output_file, "w", encoding="utf-8") as fout:
        for item in data:
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
    return


def get_entity_list(token_list: list):
    # 此处只识别 BI?E 的模式
    results = []
    last_word = ""
    last_start, last_end = -1, -1
    last_type = ""
    for token_state in token_list:
        entity = token_state["entity"].split("-")
        pos, tpe = entity[0], entity[1]
        assert token_state["start"] >= last_end
        word = token_state["word"]
        if word.startswith("##"):
            word = word[2:]
        if last_word == "":
            if pos == "B":
                last_word = word
                last_start = token_state["start"]
                last_end = token_state["end"]
                last_type = tpe
            elif pos == "S":
                results.append({
                    "entity": token_state["word"],
                    "type": tpe,
                    "start": token_state["start"],
                    "end": token_state["end"]
                })
        else:
            if (pos == "I" or pos == "E") and last_end == token_state["start"]:
                last_word += word
                last_end = token_state["end"]
                
                if pos == "E":
                    results.append({
                        "entity": last_word,
                        "type": last_type,
                        "start": last_start,
                        "end": last_end
                    })
                    last_word = ""
                    last_start, last_end = -1, -1
    return results


def get_idx_list_dict(entity_list: list):
    key2ids = {}
    for idx, entity in enumerate(entity_list):
        tpe = entity["type"]
        if tpe in key2ids:
            key2ids[tpe].append(idx)
        else:
            key2ids[tpe] = [idx]
    return key2ids


def n_gram_match(a: str, b: str, min_length: int=2):
    if a == b:
        return True
    if len(a) < min_length:
        return False
    for i in range(len(a) - min_length + 1):
        if a[i: i + min_length] in b:
            return True
    return False


def split_sentence(sent: str, ner_list: list):
    spans = []
    idx = []
    last_end, k = 0, 0
    for item in ner_list:
        st, ed = item["start"], item["end"]
        if st == last_end:
            spans.append((st, ed))
            last_end = ed
            idx.append(k)
            k = k + 1
        elif st > last_end:
            spans.extend([(last_end, st), (st, ed)])
            last_end = ed
            idx.append(k + 1)
            k = k + 2
        else:
            print("there seems to be something wrong")
    word_cuts = [sent[s[0]: s[1]] for s in spans]
    if len(spans) > 0:
        ed = spans[-1][1]
        word_cuts.append(sent[ed:])

    return word_cuts, idx


def swap_ner(summ: str, summ_ner_list: list, text_ner_list: list, max_replace_num: int=4, min_length: int=2):
    text_key2ids = get_idx_list_dict(text_ner_list)
    word_cuts, idx = split_sentence(summ, summ_ner_list)
    replace_cnt = 0
    for i in range(len(idx)):
        word_to_replace, word_type = summ_ner_list[i]["entity"], summ_ner_list[i]["type"]
        text_words = [text_ner_list[j]["entity"] for j in text_key2ids.get(word_type, [])]
        different_text_words = list(filter(lambda s: not n_gram_match(word_to_replace, s, min_length), text_words))
        if len(different_text_words) == 0:
            continue
        word_chosen = random.choice(different_text_words)
        word_cuts[idx[i]] = word_chosen
        replace_cnt += 1
        if replace_cnt >= max_replace_num:
            break
    summ_replaced = "".join(word_cuts)

    return summ_replaced, replace_cnt
