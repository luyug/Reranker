# AgAsk BERT Reranker

### Step 1 - Create feature file

First create the feature JSON file:

```bash
python helpers/build_train_from_ranking_agask.py \
  --tokenizer_name /Users/koo01a/Documents/Reranker/pt-bert-large-msmarco \
  --qrel qrels/qrel-known_item-passage.tsv \
  --json_dir feature_json \
  --query_collection queries/agask_questions-train_not50.csv \
  --doc_collection docs/agask_docs.csv
```

This will create `feature_json/agask_docs.features.json` which is used for training. 


### Setep 2 - train/eval model

```zsh
python run_agask.py --output_dir agask_model --model_name_or_path /Users/koo01a/doc/Reranker/pt-bert-large-msmarco --do_train --max_len 512  --train_group_size 8 --per_device_train_batch_size 8 --per_device_eval_batch_size 64 --train_dir feature_json
```