# Reranker
Reranker is a lightweight, effective and efficient package for training and deploying deep languge model reranker in information retrieval (IR), question answering (QA) and many other natural language processing (NLP) pipelines. 
The training procedure follows our ECIR paper [Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline](https://arxiv.org/abs/2101.08751) using a localized constrastive esimation (LCE) loss.

Reranker speaks Huggingface🤗 language! This means that you instantly get all state-of-the-art pre-trained models as soon as they are ported to HF transformers. You also get the familiar model and trainer interfaces.

### Stae of the Art Performance.
Reranker has two submissions to MS MARCO document leaderboard. Each got 1st place, advancing the SOTA!

| Date  | Submission Name |  Dev MRR@100 | Eval MRR@100  |
|---|---|---|---|
| 2021/01/20 | LCE loss + HDCT (ensemble)  | 0.464 | 0.405|
| 2020/09/09 | HDCT top100 + BERT-base FirstP (single) | 0.434 | 0.382 |

### Features
- Training rerankers from the state-of-the-art pre-trained language models like BERT, RoBERTa and ELECTRA.
- The state-of-the-art reranking performance with our LCE loss based training pipeline.
- GPU memory optimizations: Loss Parallelism and Gradient Cache which allow training of larger model.
- Faster training
    - Distributed Data Parallel (DDP) for multi GPUs. 
    - Automatic Mixed Precision (AMP) training and inference with up to 2x speedup!
- Break CPU RAM limitation by memory mapping datasets with `pyarrow` through `datasets` package interface.
- Checkpoint interoperability with Hugging Face `transformers`.

### Design Philosophy
The library is designed to be dedicated for text reranking modeling, training and testing. This helps us keep the code concise and focus on a more specific task. 

Under the hood, Reranker provides a thin layer of wrapper over Huggingface libraries. Our model wraps `PreTrainedModel` and our trainer sub-class Huggingface `Trainer`. You can then work with the familiar interfaces. 

## Installation and Dependencies
Reranker uses Pytorch, Huggingface Transformers and Datasets.  Install with the following commands,
```
git clone https://github.com/luyug/Reranker.git
cd Reranker
pip install .
```
Reranker has been tested with `torch==1.6.0, transformers==4.2.0, datasets==1.1.3`.

For development, install as editable,
```
pip install -e .
```

## Workflow
### Inference (Reranking)
The easiest way to do inference is to use one of our uploaded [trained checkpoints](https://huggingface.co/Luyu) with `RerankerForInference`.
```
from reranker import RerankerForInference
rk = RerankerForInference.from_pretrained("Luyu/bert-base-mdoc-bm25")  # load checkpoint

inputs = rk.tokenize('weather in new york', 'it is cold today in new york', return_tensors='pt')
score = rk(inputs).logits
``` 
### Training
For training, you will need a model, a dataset and a trainer. Say we have parsed arguments into
 `model_args`, `data_args` and `training_args` with `reranker.arguments`. First, 
initialize the reranker and tokenizer from one of 
[pre-tained language models](https://huggingface.co/transformers/pretrained_models.html) from Hugging Face.
For example, let's use RoBERTa by loading `roberta-base`.
```
from reranker import Reranker 
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = Reranker.from_pretrained(model_args, data_args, training_args, 'roberta-base')
```
Then create the dataset,
```
from reranker.data import GroupedTrainDataset
train_dataset = GroupedTrainDataset(
    data_args, data_args.train_path, 
    tokenizer=tokenizer, train_args=training_args
)
```
Create a trainer and train,
```
from reranker import RerankerTrainer
trainer = RerankerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=GroupCollator(tokenizer),
    )
trainer.train()
```
See full examples in our [examples](examples).
## Examples
[MS MARCO Document Ranking with Reranker](examples/msmarco-doc) 

*More to come*

## Large Models
### Loss Paralellism
We support computing a query's LCE loss with multiple GPUs with flag `--collaborative`. Note that a group size (pos + neg) 
not divisible by number of GPUs may incur undefined behaviours.
You will typically want to use it with gradient accumulation steps greater than one. 

*Detailed instruction ot be added.*

### Gradient Cache 
*Experimental*    We provide subclasses `RerankerDC` and `RerankerDCTrainer`. In the MS MARCO example, You can use them with `--distance_cahce` argument to activate gradient caching with respect to computed unnormalized distance. This allows potentially training with unlimited number of negatives beyond GPU memory limitation up to numerical precision. 
The method is described in our preprint [Scaling Deep Contrastive Learning Batch Size with Almost Constant Peak Memory Usage](https://arxiv.org/abs/2101.06983).


*Detailed instruction to be added.*

## Helpers
We provide a few helpers in the helper directory for data formatting,
### Score Formatting
- `score_to_marco.py` turns a raw score txt file into MS MARCO format.
- `score_to_tein.py` turns a raw score txt file into trec eval format.

For example,
```
python score_to_tein.py --score_file {path to raw score txt}
```
This generates a trec eval format file in the same directory as the raw score file. 
## Data Format
Reranker core utilities (batch training, batch inference) expect processed and tokenized text in token id format. 
This means pre-processing should be done beforehand, e.g. with BERT tokenizer.

### Training Data
Training data is grouped by query into a json file where each line has a query, its corresponding positives and sampled negatives.
```
{
    "qry": {
        "qid": str,
        "query": List[int],
    },
    "pos": List[
        {
            "pid": str,
            "passage": List[int],
        }
    ],
    "neg": List[
        {
            "pid": str,
            "passage": List[int]
        }
    ]
}
```
Training data is handled by class `reranker.data.GroupedTrainDataset`.
### Inference (Reranking) Data
Inference data is grouped by query document(passage) pairs. Each line is a json entry to be rereanked (scored).
```
{
    "qid": str,
    "pid": str,
    "qry": List[int],
    "psg": List[int]
}
```
To speed up postprocessing, we currently take an additional tsv specifying text ids,
```
qid0     pid0
qid0     pid1
...
```
The ordering in the two files are expected to be the same.

Inference data is handled by class `reranker.data.PredictionDataset`.
### Result Scores
Scores are stored in a tsv file with columns corresponding to qid, pid and score.
```
qid0     pid0     s0
qid0     pid1     s1
...
```
You can post-process it with our helper scirpt into MS MARCO format or TREC eval format.


## Contribution
We welcome contribution to the package, either adding new dataset interface or new models.

## Contact
You can reach me by email `luyug@cs.cmu.edu`. As a 2nd year master, I get busy days from time to time and may not reply very promptly. Feel free to ping me if you don't get replies.

## Citation
If you use Reranker in your research, please consider citing our [ECIR paper](https://arxiv.org/abs/2101.08751),

```
@inproceedings{gao2021lce,
               title={Rethink Training of BERT Rerankers in Multi-Stage Retrieval Pipeline}, 
               author={Luyu Gao and Zhuyun Dai and Jamie Callan},
               year={2021},
               booktitle={The 43rd European Conference On Information Retrieval (ECIR)},
      
}
```

For the gradient cache utility, consider citing our [preprint](https://arxiv.org/abs/2101.06983),
```
@misc{gao2021scaling,
      title={Scaling Deep Contrastive Learning Batch Size with Almost Constant Peak Memory Usage}, 
      author={Luyu Gao and Yunyi Zhang},
      year={2021},
      eprint={2101.06983},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License
Reranker is currently licensed under CC-BY-NC 4.0.



