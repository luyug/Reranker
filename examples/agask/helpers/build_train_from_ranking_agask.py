# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import logging
from argparse import ArgumentParser
from transformers import AutoTokenizer
import json
import os
import csv
from collections import defaultdict
import datasets
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)


def read_qrel():
    qrel = defaultdict(list)
    rankings = defaultdict(list)
    with open(args.qrel, 'rt', encoding='utf8') as fh:
        for line in fh:
            topicid, _, docid, rel = line.split()
            if int(rel) > 0:
                qrel[topicid].append(docid)
            else:
                rankings[topicid].append(docid)
    return qrel, rankings


def main(args):
    qrel, rankings = read_qrel()

    logging.info(f"Reading documents from {args.doc_collection}")
    columns = ['did', 'url', 'title', 'body']
    collection = args.doc_collection
    collection = datasets.load_dataset(
        'csv',
        data_files=collection,
        column_names=columns,
        delimiter=',',
        ignore_verifications=True,
    )['train']

    logging.info(f"Reading queries from {args.query_collection}")
    qry_collection = args.query_collection
    qry_collection = datasets.load_dataset(
        'csv',
        data_files=qry_collection,
        column_names=['qid', 'qry'],
        delimiter=',',
        ignore_verifications=True,
    )['train']

    doc_map = {x['did']: idx for idx, x in enumerate(collection)}
    qry_map = {str(x['qid']): idx for idx, x in enumerate(qry_collection)}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    out_file = args.doc_collection
    if out_file.endswith('.tsv') or out_file.endswith('.txt') or out_file.endswith('.csv'):
        out_file = out_file[:-4]
    out_file = os.path.join(args.json_dir, os.path.split(out_file)[1])
    out_file = out_file + '.group.json'

    queries = list(rankings.keys())
    with open(out_file, 'w') as f:
        for qid in tqdm(queries):
            # pick from top of the full initial ranking
            negs = rankings[qid]
            # shuffle if random flag is on
            if args.random:
                random.shuffle(negs)
            # pick n samples
            negs = negs[:args.n_sample]

            neg_encoded = []
            for neg in negs:
                idx = doc_map[neg]
                item = collection[idx]
                did, url, title, body = (item[k] for k in columns)
                url, title, body = map(lambda v: v if v else '', [url, title, body])
                encoded_neg = tokenizer.encode(
                    url + tokenizer.sep_token + title + tokenizer.sep_token + body,
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True
                )
                neg_encoded.append({
                    'passage': encoded_neg,
                    'pid': neg,
                })
            pos_encoded = []
            for pos in qrel[qid]:
                idx = doc_map[pos]
                item = collection[idx]
                did, url, title, body = (item[k] for k in columns)
                url, title, body = map(lambda v: v if v else '', [url, title, body])
                encoded_pos = tokenizer.encode(
                    url + tokenizer.sep_token + title + tokenizer.sep_token + body,
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True
                )
                pos_encoded.append({
                    'passage': encoded_pos,
                    'pid': pos,
                })
            q_idx = qry_map[qid]
            query_dict = {
                'qid': qid,
                'query': tokenizer.encode(
                    qry_collection[q_idx]['qry'],
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True),
            }
            item_set = {
                'qry': query_dict,
                'pos': pos_encoded,
                'neg': neg_encoded,
            }
            f.write(json.dumps(item_set) + '\n')

    print(f"Results written to {out_file}")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tokenizer_name', required=True)
    parser.add_argument('--truncate', type=int, default=512)

    parser.add_argument('--n_sample', type=int, default=100)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--json_dir', required=True)

    parser.add_argument('--qrel', required=True)
    parser.add_argument('--query_collection', required=True)
    parser.add_argument('--doc_collection', required=True)
    args = parser.parse_args()

    main(args)
