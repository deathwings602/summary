INPUT_FILE=/home/huangshuhong/huangshuhong/data/LCSTS_stage1/train.jsonl
OUTPUT_FILE=/home/huangshuhong/huangshuhong/data/LCSTS_stage2/train.jsonl

python ./code/get_ner.py --input-file $INPUT_FILE --output-file $OUTPUT_FILE