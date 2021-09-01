import argparse
import os
from pyserini.search import SimpleSearcher
from tqdm import tqdm


def pretty_print_results(self, results, num_results=10, label=""):
    for i in range(0, num_results):
        print(f'{label} - {i + 1:2} {results[i].metadata["docid"]:15} {results[i].score:.5f}')  # {results[i].text}')


def load_queries(f):
    for line in f:
        if len(line.strip()) == 0:
            continue
        qid, text = line.strip().split(",", 1)
        yield qid, text


def search(searcher, query, depth):
    hits = searcher.search(query, depth)
    rank_list = []
    for rank, hit in enumerate(hits):
        rank_list.append({"query": query, "doc": hit.docid, "rank": rank + 1, "score": hit.score})
    return rank_list


def run_queries(index_dir, queries, collection_file, run_file, depth=1000):
    searcher = SimpleSearcher(index_dir)
    collection = {}

    with open(run_file, 'w') as out_file:
        for qid, query_str in tqdm(queries.items(), desc=f"Running {len(queries)} queries"):
            for r in search(searcher, query_str, depth):
                out_file.write(f"{qid}\tQ0\t{r['doc']}\t{r['rank']}\t{r['score']:.4f}\t{run_file}\n")
                collection[r['doc']] = searcher.doc(r['doc']).contents().replace("\n", " ")

    with open(collection_file, 'w') as out_file:
        for docid, text in collection.items():
            out_file.write(f"{docid}\t{text}\n")

    print(f"Seacrch results written to {run_file}; collection written to {collection_file}")


if __name__ == '__main__':

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)


    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=dir_path, required=True)
    parser.add_argument('--queries', type=argparse.FileType('r'), required=True)
    parser.add_argument('-d', '--depth', type=int, default=1000, help='Retrieve up to rank depth')
    parser.add_argument('collection', help="File to write collection")
    parser.add_argument('ranking', help="File to write ranking")

    args = parser.parse_args()

    qs = dict(load_queries(args.queries))

    run_queries(args.index_dir, qs, args.collection, args.ranking)
