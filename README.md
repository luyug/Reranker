# Reranker
Reranker is a lightweight, effective and efficient package for training and deploying deep languge model reranker in information retrieval (IR), question answering (QA) and many other natural language processing (NLP) pipelines. 
The training procedure follows our ECIR paper using a localized constrastive esimation (LCE) loss.

Reranker speaks Huggingface🤗  language! This means that you instantly get all SOTA pre-trained models as soon as they are ported to HF transformers. You also get the familiar model and trainer interfaces！

### Stae of the Art Performance.
Reranker has two submissions to MS MARCO document leaderboard. Each got 1st place, advancing the SOTA!

| Date  | Submission Name |  Dev MRR@100 | Eval MRR@100  |
|---|---|---|---|
| 2021/01/20 | LCE loss + HDCT (ensemble)  | 0.464 | 0.405|
| 2020/09/09 | HDCT top100 + BERT-base FirstP (single) | 0.434 | 0.382 |

### Features
- Training rerankers from pre-trained the state-of-the-art language models like BERT, RoBERTa and ELECTRA.
- State-of-the-art reranking performance with our LCE loss based training pipeline.
- GPU memory optimizations: Loss Parallelism and Gradient Cache which allow training of larger model.
- Faster training
    - Distributed Data Parallel (DDP) for multi GPUs. 
    - Automatic Mixed Precision (AMP) training and inference with up to 2x speedup!
- Break CPU RAM limitation by memory mapping datasets with `pyarrow ` through `datasets` package interface.
- Checkpoint interoperability with HF `transformers`.

### Design Philosophy
The library is designed to be dedicated for text reranking modeling, training and testing. This helps us keep the code concise and focus on a more specific task. 

Under the hood, Reranker provides a thin layer of wrapper over Huggingface libraries. Our model wraps `PreTrainedModel` and our trainer sub-class Huggingface `Trainer`. You can then work with the familiar interfaces. 

## Installation and Depencies
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
Here is a code snippet to start reranker training on MS MARCO with roberta-base 
```
from reranker import Reranker, RerankerTrainer
from reranker.arguments import ModelArguments, DataArguments, \
    RerankerTrainingArguments as TrainingArguments
from reranker.data import GroupedTrainDataset, GroupCollator

# get arguments
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# create tokenizer and model
tokenizer = AutoTokenizer.from_pretrained('roberta-base')
model = Reranker.from_pretrained(model_args, data_args, training_args, 'roberta-base')

# create train dataset
train_dataset = GroupedTrainDataset(data_args, data_args.train_path, tokenizer=tokenizer, train_args=training_args)

# initialize trainer and train
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
- `score_to_marco.py` turns a raw score txt file into MS MARCO format.
- `score_to_tein.py` turns a raw score txt file into trec eval format.

*Detailed instructions to be added.*

## Data Format
Reranker core utitlities expect precoessed and tokenized text in token id form. 
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
If you use Reranker in your research, please consider citing our ECIR paper,

```
@inproceedings{gao2021lce,
               title={Rethink Training of BERT Rerankersin Multi-Stage Retrieval Pipeline}, 
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



