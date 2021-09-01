# AgAsk BERT Reranker

First create the feature JSON file:

```bash
python helpers/build_train_from_ranking_agask.py --tokenizer_name /Users/koo01a/Documents/Reranker/pt-bert-large-msmarco --qrel qrels/qrel-known_item-passage.tsv --json_dir feature_json --query_collection queries/agask_queries.csv --doc_collection docs/agask_docs.csv
```

