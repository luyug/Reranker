from argparse import ArgumentParser
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from timeit import default_timer as timer

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc="loading collection...."):
            docid, text = line.strip().split("\t", 1)
            collection[docid] = text
    return collection


def load_queries(query_path):
    query = {}
    with open(query_path, 'r') as f:
        for line in tqdm(f, desc="loading query...."):
            if len(line.strip()) == 0:
                continue
            qid, text = line.strip().split(",", 1)
            query[qid] = text
    return query


def load_run(run_path, run_type='msmarco'):
    run = {}
    with open(run_path, 'r') as f:
        for line in tqdm(f, desc="loading run...."):
            if run_type == 'msmarco':
                qid, docid, score = line.strip().split()
            elif run_type == 'trec':
                qid, _, docid, rank, score, _ = line.strip().split()
            qid = qid
            docid = docid
            if qid not in run.keys():
                run[qid] = []
            run[qid].append(docid)
    return run


def batch_test_iter(queries, texts, batch_size):
    assert len(queries) == len(texts)
    for i in range(0, len(queries), batch_size):
        yield queries[i: i + batch_size], texts[i: i + batch_size]


def rerank_one_query(query, texts, tokenizer, model, batch_size):
    scores = []
    for batch_query, batch_pass in batch_test_iter([query] * len(texts), texts, batch_size):
        inputs = tokenizer(batch_query,
                           batch_pass,
                           add_special_tokens=True,
                           return_token_type_ids=True,
                           max_length=512, truncation=True, padding=True, return_tensors='pt').to(DEVICE)
        with torch.no_grad():
            logits = model(inputs["input_ids"],
                           attention_mask=inputs["attention_mask"],
                           token_type_ids=inputs["token_type_ids"])[0]
            score = torch.softmax(logits, dim=1)
            scores.extend(score.detach().cpu().numpy()[:, 1])
    return scores


def main(args):
    collection = load_collection(args.collection_file)
    queries = load_queries(args.query_file)
    run = load_run(args.run_file, run_type=args.run_type)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path,
                                                               cache_dir=args.cache_dir).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path,
                                              use_fast=True,
                                              cache_dir=args.cache_dir)

    lines = []
    total_ranking_time = 0
    if len(run.keys()) < len(queries.keys()):
        qids = run.keys()
    else:
        qids = queries.keys()

    for qid in tqdm(qids):
        query = queries[qid]
        texts = [collection[pid] for pid in run[qid]]
        ranking_start = timer()

        scores = rerank_one_query(query, texts[:args.cut_off], tokenizer, model, batch_size=args.batch_size)
        zipped_lists = zip(scores, run[qid][:args.cut_off])
        sorted_pairs = sorted(zipped_lists, reverse=True)

        ranking_end = timer()
        total_ranking_time += (ranking_end - ranking_start)

        for i in range(len(sorted_pairs)):
            score, docid = sorted_pairs[i]
            if args.run_type == 'msmarco':
                lines.append(str(qid) + "\t" + str(docid) + "\t" + str(i + 1) + "\n")
            else:
                lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(i + 1) + " " + str(
                    score) + " " + "bert" + "\n")
            last_score = score
            last_rank = i

        # add the rest of ranks below cut_off, we don't need to re-rank them.
        for docid in run[qid][last_rank + 1:]:
            last_score -= 1
            last_rank += 1
            if args.run_type == 'msmarco':
                lines.append(str(qid) + "\t" + str(docid) + "\t" + str(last_rank + 1) + "\n")
            else:
                lines.append(str(qid) + " " + "Q0" + " " + str(docid) + " " + str(last_rank + 1) + " " + str(
                    last_score) + " " + "bert" + "\n")

    print("passage re-ranking time: %.1f ms" % (1000 * total_ranking_time / len(qids)))

    with open(args.output_path, "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--run_file', required=True, type=str)
    parser.add_argument('--collection_file', required=True, type=str)
    parser.add_argument('--query_file', required=True, type=str)
    parser.add_argument('--run_type', type=str, default='trec')
    parser.add_argument('--model_name_or_path', type=str, default='nboost/pt-bert-large-msmarco')
    parser.add_argument('--tokenizer_name_or_path', type=str, default='bert-large-uncased')
    parser.add_argument('--output_path', required=True, type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--cut_off', type=int, default=1000)
    parser.add_argument('--cache_dir', type=str, default='cache')
    args = parser.parse_args()
    main(args)

# python3 bert_reranker.py \
# --run_file runs/DL2019/bm25-top1000-dl2019-pass.txt \
# --collection_file collection.tsv \
# --query_file queries/DL2019-queries.tsv \
# --model_name_or_path nboost/pt-bert-large-msmarco \
# --tokenizer_name_or_path nboost/pt-bert-large-msmarco \
# --batch_size 128 \
# --run_type trec \
# --output_path bert-large-bm25-top500-dl2019-pass.res \
# --cut_off 500
