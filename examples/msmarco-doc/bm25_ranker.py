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


def search(index_dir, queries, collection_file, run_file):
    searcher = SimpleSearcher(index_dir)

    collection = {}

    with open(run_file, 'w') as out_file:
        for query in tqdm(queries, desc=f"Running {len(queries)} queries"):
            hits = searcher.search(query)
            for hit in hits:
                out_file.write(f"{query} {hit.docid} {hit.score}\n")
                collection[hit.docid] = searcher.doc(hit.docid).contents()

    with open(collection_file, 'w') as out_file:
        for docid, text in collection.items():
            out_file.write(f"{docid}\t{text}\n")


if __name__ == '__main__':

    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)


    parser = argparse.ArgumentParser()
    parser.add_argument('--index_dir', type=dir_path, required=True)
    parser.add_argument('--queries', type=argparse.FileType('r'), required=True)
    parser.add_argument('collection', help="File to write collection")
    parser.add_argument('ranking', help="File to write ranking")

    args = parser.parse_args()

    qs = dict(load_queries(args.queries))

    search(args.index_dir, qs, args.collection, args.ranking)
