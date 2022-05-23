import json
import argparse
import os

import numpy as np

from tqdm import tqdm


def get_args():
	parser = argparse.ArgumentParser()

	# input data
	parser.add_argument('--data-dir', type=str, default='/home/zhaoxinhao/data2/cpm1/data')
	parser.add_argument('--dataset', type=str, default='LCSTS')
	parser.add_argument('--train-file-name', type=str, default='train.jsonl')
	parser.add_argument('--dev-file-name', type=str, default='dev.jsonl')
	parser.add_argument('--test-file-name', type=str, default='test_private.jsonl')

	# output
	parser.add_argument('--output-dir', type=str, default='/home/zhaoxinhao/data2/cpm1/data')

	return parser.parse_args()


def main():
	args = get_args()

	file_dir = os.path.join(args.data_dir, args.dataset)

	train_data = set()
	train_map = {}
	with open(os.path.join(file_dir, args.train_file_name), encoding='utf8') as fin:
		for line in tqdm(fin):
			data = json.loads(line)
			train_data.add((tuple(data['summary']), tuple(data['text'])))
			train_map[tuple(data['summary'])] = data['text']

	output_dir = os.path.join(args.output_dir, args.dataset)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	with open(os.path.join(file_dir, args.dev_file_name), encoding='utf8') as fin, open(os.path.join(args.output_dir, args.dataset, args.dev_file_name + '.dedup_all'), 'w') as fout:
		for i, line in tqdm(enumerate(fin)):
			data = json.loads(line)
			rig_tokens = (tuple(data['summary']), tuple(data['text']))
			if rig_tokens in train_data:
				print(i, rig_tokens)
			else:
				if tuple(data['summary']) in train_map:
					fout.write('---' + train_map[tuple(data['summary'])] + '---' + line)
				else:
					fout.write(line)

	with open(os.path.join(file_dir, args.test_file_name), encoding='utf8') as fin, open(os.path.join(args.output_dir, args.dataset, args.test_file_name + '.dedup_all'), 'w') as fout:
		for i, line in tqdm(enumerate(fin)):
			data = json.loads(line)
			rig_tokens = (tuple(data['summary']), tuple(data['text']))
			if rig_tokens in train_data:
				print(i, rig_tokens)
			else:
				if tuple(data['summary']) in train_map:
					fout.write('---' + train_map[tuple(data['summary'])] + '---' + line)
				else:
					fout.write(line)


if __name__ == '__main__':
	main()
