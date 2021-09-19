# CausalFSED

- An implementation for EMNLP 2021 short paper [``Honey or Poison? Solving the Trigger Curse in Few-shot Event Detection via Causal Intervention``](https://arxiv.org/abs/2109.05747)

## Quick links
* [Environment](#Environment)
* [Dataset](#Dataset)
* [Model Training](#Model-Training)
* [Model Evaluation](#Model-Evaluation)

### Environment

```bash
conda create -n causalfsed python=3.7.3
conda activate causalfsed
pip install -r requirements.txt
```


### Dataset
Custom dataset can be putted in data folder:

```text
data/custom
├── sentence.json
├── test.json
├── train.json
├── dev.json
└── getquery.py
```

sentence.json is data file, and the format is List and each element is a Dict. Each Dict contains `words` and `event` fields, in which `text` is the list of tokens, and `event` is the list of event mentions.


```text
[
    {
        "words": [token1,token2,...],
        "event": [
            [
                [eventtype1, {"text": trigger word1, "start": startindex1, "end": endindex2],
                [eventtype2, {"text": trigger word2, "start": startindex2, "end": endindex2],
                ...
            ]
    },
    ...
]
```
train/dev/test.json record the index of sentence for train/dev/test set, and the format is Dict. The key of Dict is  `event type` and the value of Dict is a List which contains the index of sentence in sentecen.json.
```text
train.json
{
    eventtype1: [sentenceindex1, sentenceindex2, ...], 
    eventtype2: [sentenceindex1, sentenceindex2, ...],
    ...
}
```
The file getquery.py is used to obtain ambigous triggers for dev set which are used in training.
```bash
cd data/custom
python getquery.py
```
For ACE2005, KBP2017 and MAVEN dataset, the files data/ace2005/acepreprocess.py, data/maven/mavenpreprocess.py and data/kbp2017/kbppreprocess.py can be used to convert some other format to this format.

### Model Training
run:
```bash
python main.py -model FSCausal -metric proto -dataset ace -cuda 0
```
+ -model can be `FSBase` and `FSCausal`
+ -metric can be `proto` and `relation`
+ -dataset can be `ace`, `kbp`, `maven` and `custom`
+ -cudaid is the gpu id.

where proto is Prototypical Network and relation is Relation Network.

The model checkpoint will be saved in tmp/dataset_name/...

### Model Evaluation
just add -t:
```bash
python main.py -model FSCausal -metric proto -dataset ace -cuda 0 -t
```

