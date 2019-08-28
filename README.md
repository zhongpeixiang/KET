The PyTorch code for paper: Knowledge-Enriched Transformer for Emotion Detection in Textual Conversations

## Steps for Each Dataset

- Download data: download data to respective foler in `./data/`, install [Magnitude Medium GloVe](https://github.com/plasticityai/magnitude) for pretrained word embedding.
- Preprocess data: run `preprocess.py` to process `csv` files to `pkl` data.
- Download [ConceptNet](https://github.com/commonsense/conceptnet5/wiki/Downloads) and [NRC_VAD](https://saifmohammad.com/WebPages/nrc-vad.html).
- Preprocess ConceptNet and NRC_VAD: run `preprocess_conceptnet.py` and `preprocess_NRC_VAD.py`.
- Model training and evluation: run `train.py`. 

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