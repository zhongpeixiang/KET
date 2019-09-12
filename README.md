The PyTorch code for paper: Knowledge-Enriched Transformer for Emotion Detection in Textual Conversations

The model is largely based on [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## Steps

- Download data: download data to respective foler in `./data/`: `EC`, `DD`, `MELD`, `EmoryNLP`, and `IEMOCAP`. 
- Install [Magnitude Medium GloVe](https://github.com/plasticityai/magnitude) for pretrained word embedding.
- Preprocess data: run `preprocess.py` to process `csv` or `pkl` (IEMOCAP) files into `pkl` data.
- Download [ConceptNet](https://github.com/commonsense/conceptnet5/wiki/Downloads) and [NRC_VAD](https://saifmohammad.com/WebPages/nrc-vad.html).
- Preprocess ConceptNet and NRC_VAD: run `preprocess_conceptnet.py` and `preprocess_NRC_VAD.py`.
- Model training: run `train.py`. 
- Model evaluation: run `train.py` with `test_mode` set.

## Citing
If you find this repo useful, please cite
```
@inproceedings{zhong2019knowledge,
    title={Knowledge-Enriched Transformer for Emotion Detection in Textual Conversations},
    author={Peixiang, Zhong and Di, Wang and Chunyan, Miao},
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP)",
    year={2019}
}
```