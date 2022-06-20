---
layout: post
title:  "OntoNotes v5.0 for NER"
date:   2022-06-20 05:03:26 +0530
---
## Introduction

**Objective**: To explore `[OntoNotes]` dataset for NER task and convert it to `CONLL` format.

- Project: https://catalog.ldc.upenn.edu/LDC2013T19
- Release Year: 2013
- Data Sources: telephone talks, newswire, newsgroups, weblogs, religious text, etc.
- Format: Penn Treebank
- Languages: English, Chinese, Arabic 
- Usage: NER, POS, Coreference resolution

Following steps that we will take into action for achiving our objective.
- load this dataset from huggingface itself.
- convert to conll format and save to disk. 

At the end we will have train.conll, test.conll and validate.conll files.

## Implementation

### import libraries

Importing just huggingface's `datasets` library.


```python
import datasets
```

Assert the library version to make sure we are using that as expected versions. 


```python
assert datasets.__version__ == '2.3.2'
```

### load dataset

Loading a well prepared `Ontonotes v5` dataset from huggingface itself.


```python
dset = datasets.load_dataset("conll2012_ontonotesv5", "english_v4")
```

    Reusing dataset conll2012_ontonotesv5 (/home/djagatiya/.cache/huggingface/datasets/conll2012_ontonotesv5/english_v4/1.0.0/c541e760a5983b07e403e77ccf1f10864a6ae3e3dc0b994112eff9f217198c65)
    100%|██████████| 3/3 [00:00<00:00, 147.00it/s]


let's see how many samples which we have ? A dataset is already splitted out into train/test/validate. 

We have `1940` samples in training and `222` for testing.
 
Here `Sample` mean `Document` and single document does have multiple sentences.


```python
dset
```




    DatasetDict({
        train: Dataset({
            features: ['document_id', 'sentences'],
            num_rows: 1940
        })
        validation: Dataset({
            features: ['document_id', 'sentences'],
            num_rows: 222
        })
        test: Dataset({
            features: ['document_id', 'sentences'],
            num_rows: 222
        })
    })




```python
train_set = dset['train']
```


```python
print(train_set)
```

    Dataset({
        features: ['document_id', 'sentences'],
        num_rows: 1940
    })


### dataset understanding

This dataset is being used for multiple purpose. like NER (Named Entity recognization), POS (Part of speech tagging) and Coreference resolution.

Let have look for features, here dataset have words, pos_tags, parse_tree, word_senses, named_entities featues.


```python
train_set.features
```




    {'document_id': Value(dtype='string', id=None),
     'sentences': [{'part_id': Value(dtype='int32', id=None),
       'words': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
       'pos_tags': Sequence(feature=ClassLabel(num_classes=49, names=['XX', '``', '$', "''", ',', '-LRB-', '-RRB-', '.', ':', 'ADD', 'AFX', 'CC', 'CD', 'DT', 'EX', 'FW', 'HYPH', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NFP', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB'], id=None), length=-1, id=None),
       'parse_tree': Value(dtype='string', id=None),
       'predicate_lemmas': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
       'predicate_framenet_ids': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
       'word_senses': Sequence(feature=Value(dtype='float32', id=None), length=-1, id=None),
       'speaker': Value(dtype='string', id=None),
       'named_entities': Sequence(feature=ClassLabel(num_classes=37, names=['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME', 'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE'], id=None), length=-1, id=None),
       'srl_frames': [{'verb': Value(dtype='string', id=None),
         'frames': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)}],
       'coref_spans': Sequence(feature=Sequence(feature=Value(dtype='int32', id=None), length=3, id=None), length=-1, id=None)}]}



As we know that a dataset made of document and single document has multiplt sentences. An each sentences has one feature called `named_entities` inside it sub featues called `names`. that contain the actual entity name.


```python
train_set.features['sentences'][0]['named_entities'].feature
```




    ClassLabel(num_classes=37, names=['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME', 'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE'], id=None)




```python
names = train_set.features['sentences'][0]['named_entities'].feature.names
print(names)
```

    ['O', 'B-PERSON', 'I-PERSON', 'B-NORP', 'I-NORP', 'B-FAC', 'I-FAC', 'B-ORG', 'I-ORG', 'B-GPE', 'I-GPE', 'B-LOC', 'I-LOC', 'B-PRODUCT', 'I-PRODUCT', 'B-DATE', 'I-DATE', 'B-TIME', 'I-TIME', 'B-PERCENT', 'I-PERCENT', 'B-MONEY', 'I-MONEY', 'B-QUANTITY', 'I-QUANTITY', 'B-ORDINAL', 'I-ORDINAL', 'B-CARDINAL', 'I-CARDINAL', 'B-EVENT', 'I-EVENT', 'B-WORK_OF_ART', 'I-WORK_OF_ART', 'B-LAW', 'I-LAW', 'B-LANGUAGE', 'I-LANGUAGE']


A document sample composed of "document_id" and it's sentences. 


```python
train_set[0]['document_id']
```




    'bc/cctv/00/cctv_0001'




```python
doc_senteces = train_set[0]['sentences']
len(doc_senteces)
```




    235




```python
doc_senteces[0]
```




    {'part_id': 0,
     'words': ['What', 'kind', 'of', 'memory', '?'],
     'pos_tags': [46, 24, 17, 24, 7],
     'parse_tree': '(TOP(SBARQ(WHNP(WHNP (WP What)  (NN kind) )(PP (IN of) (NP (NN memory) ))) (. ?) ))',
     'predicate_lemmas': [None, None, None, 'memory', None],
     'predicate_framenet_ids': [None, None, None, None, None],
     'word_senses': [None, None, None, 1.0, None],
     'speaker': 'Speaker#1',
     'named_entities': [0, 0, 0, 0, 0],
     'srl_frames': [],
     'coref_spans': []}



Our goal is to do NER so we will be using only `words` and `named_entities` featues. 


```python
print(doc_senteces[1]['words'])
```

    ['We', 'respectfully', 'invite', 'you', 'to', 'watch', 'a', 'special', 'edition', 'of', 'Across', 'China', '.']



```python
entity_names = doc_senteces[1]['named_entities']
print(entity_names)

decoded_entity_names = [names[i] for i in entity_names]
print(decoded_entity_names)
```

    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 8, 0]
    ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O']


A `save_as_conll` function is used to write dataset into file as conll format. This function will iterate every document from fullset and write pair of word and ner into file.

Format:
```
WORD \t TAG
``` 


```python
import tqdm

def save_as_conll(data_set, out_path):
    total_sentences = 0
    with open(out_path, mode='w', encoding='utf-8') as _file:
        for i, _doc in enumerate(tqdm.tqdm(data_set)):
            for _s in _doc['sentences']:
                total_sentences += 1
                for _w, _t in zip(_s['words'], _s['named_entities']):
                    _file.write(f"{_w}\t{names[_t]}\n")
                _file.write(f"\n\n")
    print("Total_sentences:", total_sentences)
```


```python
save_as_conll(train_set, "data/ontonotes/train.conll")
```

    100%|██████████| 1940/1940 [00:16<00:00, 116.70it/s]

    Total_sentences: 75187


    



```python
save_as_conll(dset['test'], "data/ontonotes/test.conll")
```

    100%|██████████| 222/222 [00:01<00:00, 121.85it/s]

    Total_sentences: 9479


    



```python
save_as_conll(dset['validation'], "data/ontonotes/validation.conll")
```

    100%|██████████| 222/222 [00:01<00:00, 123.47it/s]

    Total_sentences: 9603


    


Now we have ontonotes 5.0 dataset in conll format. which has around `70K` of training samples and `10K` samples for testing.


```
!du -k data/ontonotes/*
```

    1324	data/ontonotes/test.conll
    10052	data/ontonotes/train.conll
    1272	data/ontonotes/validation.conll


### Summary

| Split Name | # Documents | # Sentences | # Disk occupy |
| --- | --- | --- | --- |
| Train | 1940 | 75187 | 10052 KB |
| Test | 222 | 9479 | 1324 KB |
| Validate | 222 | 9603 | 1272 KB |

## Reference
- https://huggingface.co/datasets/conll2012_ontonotesv5
- https://huggingface.co/docs/datasets/loading
- https://huggingface.co/docs/datasets/access
- https://huggingface.co/docs/datasets/v2.3.2/en/package_reference/main_classes
