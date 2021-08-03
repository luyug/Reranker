# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool
import json
import datasets

columns = [
    'qid', 'query', 'did', 'url', 'title', 'body', 'unused'
]


class TopKText2Json:

    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
        self.SEP = self.tokenizer.sep_token
        self.args = args

    def encode_line(self, line):
        qid, qry, did, url, title, body = line.strip().split('\t')
        qry_encoded = self.tokenizer.encode(
            qry,
            truncation=True if args.q_truncate else False,
            max_length=args.q_truncate,
            add_special_tokens=False,
            padding=False,
        )
        doc_encoded = self.tokenizer.encode(
            url + self.SEP + title + self.SEP + body,
            truncation=True,
            max_length=args.truncate,
            add_special_tokens=False,
            padding=False
        )
        entry = {
            'qid': qid,
            'pid': did,
            'qry': qry_encoded,
            'psg': doc_encoded,
        }
        entry = json.dumps(entry)
        return entry, qid, did

    def encode_item(self, item):
        qid, qry, did, url, title, body, _ = (item[k] for k in columns)
        url, title, body = map(lambda v: v if v else '', [url, title, body])
        qry_encoded = self.tokenizer.encode(
            qry,
            truncation=True if self.args.q_truncate else False,
            max_length=self.args.q_truncate,
            add_special_tokens=False,
            padding=False,
        )
        doc_encoded = self.tokenizer.encode(
            url + self.SEP + title + self.SEP + body,
            truncation=True,
            max_length=self.args.truncate,
            add_special_tokens=False,
            padding=False
        )
        entry = {
            'qid': qid,
            'pid': did,
            'qry': qry_encoded,
            'psg': doc_encoded,
        }
        entry = json.dumps(entry)
        return entry, qid, did


if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('--file', required=True)
    parser.add_argument('--save_to', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--generate_id_to')
    parser.add_argument('--truncate', type=int, default=512)
    parser.add_argument('--q_truncate', type=int, default=16)
    args = parser.parse_args()

    data_set = datasets.load_dataset(
        'csv',
        data_files=args.file,
        column_names=columns,
        delimiter='\t',
        ignore_verifications=True
    )['train']

    top_k_2_json = TopKText2Json(args)

    with open(args.save_to, 'w') as jfile:
        # for l in text_file:
        #     json_item = method_name(args, l, tokenizer)
        all_ids = []
        if args.q_truncate < 0:
            print('queries are not truncated', flush=True)
            args.q_truncate = None
        with Pool() as p:
            all_json_items = p.imap(
                top_k_2_json.encode_item,
                tqdm(data_set),
                chunksize=100
            )
            for json_item, qry_id, doc_id in all_json_items:
                all_ids.append((qry_id, doc_id))
                jfile.write(json_item + '\n')

        if args.generate_id_to is not None:
            with open(args.generate_id_to, 'w') as id_file:
                for qry_id, doc_id in all_ids:
                    id_file.write(f'{qry_id}\t{doc_id}\n')
