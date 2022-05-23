export CUDA_VISIBLE_DEVICES=3

INPUT_FILE=/home/huangshuhong/huangshuhong/data/LCSTS/train.jsonl
OUTPUT_FILE=/home/huangshuhong/huangshuhong/data/LCSTS_stage1/train.jsonl

python ./code/ner.py --input-file $INPUT_FILE --output-file $OUTPUT_FILE --batch-size 8
