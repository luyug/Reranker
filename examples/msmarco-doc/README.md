# MS MARCO Document
This example walks through reranker LCE training and inference on MS MARCO document collection with BERT-base LM and HDCT retriever.

After downloading the data, you can also skip the steps train data building and model training by using a trained model checkpoint uploded to Hugging Face model hub. See Inference sectoin for details.

## Preparing Data
Download HDCT train rankings and dev file `hdct-marco-train.zip`, `dev.d100.tsv` from LTI server using this [link](http://boston.lti.cs.cmu.edu/appendices/TheWebConf2020-Zhuyun-Dai/rankings/) and unzip the latter.

Download the MSMARCO document ranking collection files `msmarco-doctrain-qrels.tsv.gz`, `msmarco-doctrain-queries.tsv`, `msmarco-docs.tsv` from the [official repo](https://github.com/microsoft/MSMARCO-Document-Ranking). 
Decompress the latter two.

## Building Localized Training Data from Target Retriever top Ranking
Helper script `build_train_from_ranking.py` takes a ranking file and generate training set with localized negatives. It expects a tsv with 3 columns query id, passage/document id and ranking.
```
qid  pid1  1
qid  pid2  2
...
```
Run the script with following command,
```
mkdir -p {directory to store generated json training file}
for i in $(seq -f "%03g" 0 183)
do
python helpers/build_train_from_ranking.py \
    --tokenizer_name bert-base-uncased \
    --rank_file {directory of unzipped hdct-marco-train}/${i}.txt \
    --json_dir {directory to store generated json training file} \
    --n_sample 10 \
    --sample_from_top 100 \
    --random \
    --truncate 512 \
    --qrel {path to msmarco-doctrain-qrels.tsv.gz} \
    --query_collection {path to msmarco-doctrain-queries.tsv} \
    --doc_collection {path to msmarco-docs.tsv}
done
```

## Training 
This starts training on 4 GPUs with DDP.
```
python -m torch.distributed.launch --nproc_per_node 4 run_marco.py \
  --output_dir {directory to save checkpoints} \
  --model_name_or_path  bert-base-uncased \
  --do_train \
  --save_steps 2000 \
  --train_dir {path to a train json splits from last step} \
  --max_len 512 \
  --fp16 \
  --per_device_train_batch_size 1 \
  --train_group_size 8 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 64 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --overwrite_output_dir \
  --dataloader_num_workers 8 \
```
Validatoin during training to be added. Validation over the entire dev is too expensive to do per x steps. Suggestions of alternatives are welcomed! (You can run inference during training separtely by loading saved checkpoints). After training, the last few checkpoints are usually good. 

## Inference
First build the ranking input,
```
mkdir -p {directory to save output}
python helpers/topk_text_2_json.py \
  --file {path to dev.d100.tsv} \
  --save_to {directory to save output}/all.json \
  --generate_id_to {directory to save output}/ids.tsv \
  --tokenizer bert-base-uncased \
  --truncate 512 \
  --q_truncate -1 
```
Run inference with generated input using trained model checkpoint. You can also use DDP for inference by adding `python -m torch.distributed.launch --nproc_per_node {n_gpus}`. DP is currently not supported.
```
python run_marco.py \
  --output_dir {score saving directory, not used for the moment} \
  --model_name_or_path {path to checkpoint} \
  --tokenizer_name bert-base-uncased \
  --do_predict \
  --max_len 512 \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 8 \
  --pred_path {path to prediction json} \
  --pred_id_file  {path to prediction id tsv} \
  --rank_score_path {save path of the text file of scores}
```
Or with hub model,
```
python run_marco.py \
  --output_dir {score saving directory, not used for the moment} \
  --model_name_or_path Luyu/bert-base-mdoc-hdct \
  --do_predict \
  --max_len 512 \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 8 \
  --pred_path {path to prediction json} \
  --pred_id_file  {path to prediction id tsv} \
  --rank_score_path {save path of the text file of scores}
```
Convert score to MS MARCO format. This creates a MS MARCO format score file in the same directory,
```
python {package root}/helpers/score_to_marco.py \
  --score_file {path to inference output}
```


