import os
import random
import logging
import argparse

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn.init import xavier_uniform_
from pymagnitude import Magnitude

from utils.io import load_pickle
from utils.data import Vocab, convert_examples_to_ids, create_batches, merge_splits, get_vocab_embedding, \
    filter_conceptnet, remove_KB_duplicates, get_emotion_intensity
from utils.tools import count_parameters, label_distribution_transformer

from model.transformer import make_model
from model.generator import Generator
from model.batch import flatten_examples_classification, create_batches_classification
from model.loss import SimpleLossCompute

logging.basicConfig(level = logging.INFO, \
                    format = '%(asctime)s  %(levelname)-5s %(message)s', \
                    datefmt =  "%Y-%m-%d-%H-%M-%S")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model for Context-based Emotion Classification in Conversations")
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--max_vocab_size', type=int, default=1e9)
    parser.add_argument('--context_length', type=int, default=6)
    parser.add_argument('--test_mode', action="store_true")
    
    # model
    parser.add_argument('--model_variant', type=str, default=2)
    parser.add_argument('--graph_attention_variant', type=str, default=2)
    parser.add_argument('--KB', type=str, default="conceptnet")
    parser.add_argument('--KB_percentage', type=float, default=1.0)
    parser.add_argument('--GAW', type=float, default=-1) # default -1 in paper
    parser.add_argument('--concentration_factor', type=float, default=1) # default 1 in paper
    parser.add_argument('--n_layers', type=int, default=1) # 1 layer in paper
    parser.add_argument('--d_model', type=int, default=100)
    parser.add_argument('--d_ff', type=int, default=100)
    parser.add_argument('--h', type=int, default=4)

    # training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--seed', type=int, default=1)

    args = parser.parse_args()

    device = torch.device(0)
    test_mode = args.test_mode
    dataset = args.dataset
    min_freq = args.min_freq
    max_vocab_size = int(args.max_vocab_size)

    model_variant = args.model_variant
    KB = args.KB
    KB_percentage = args.KB_percentage
    graph_attention_variant = args.graph_attention_variant
    GAW = args.GAW
    concentration_factor = args.concentration_factor
    context_length = args.context_length
    n_layers = args.n_layers
    d_model = args.d_model
    d_ff = args.d_ff
    h = args.h

    if context_length == 0:
        KB = ""
    embedding_size = d_model
    
    if dataset == "EC":
        context_length = 2
    
    # training
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    dropout = args.dropout
    seed = args.seed

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # load examples
    logging.info("Loading data...")
    train = load_pickle("./data/{0}/train.pkl".format(dataset))
    val = load_pickle("./data/{0}/val.pkl".format(dataset))
    if test_mode:
        test = load_pickle("./data/{0}/test.pkl".format(dataset))
        train = merge_splits(train, val)
        val = test
    logging.info("Number of training examples: {0}".format(len(train)))
    logging.info("Number of validation examples: {0}".format(len(val)))
    for ex in train[0][:3]:
        logging.info("Examples: {0}".format(ex))
    logging.info("Building vocab...")
    vocab = Vocab(train, min_freq, max_vocab_size)
    vocab_size = len(vocab.word2id)
    logging.info("Vocab size: {0}".format(vocab_size))
    
    # build vocab and data    
    # use pretrained word embedding
    logging.info("Loading word embedding from Magnitude...")
    home = os.path.expanduser("~")
    if embedding_size in [50, 100, 200]:
        vectors = Magnitude(os.path.join(home, "WordEmbedding/glove.twitter.27B.{0}d.magnitude".format(embedding_size)))
    elif embedding_size in [300]:
        # vectors = Magnitude(os.path.join(home, "WordEmbedding/GoogleNews-vectors-negative{0}.magnitude".format(embedding_size)))
        vectors = Magnitude(os.path.join(home, "WordEmbedding/glove.840B.{0}d.magnitude".format(embedding_size)))
    pretrained_word_embedding = get_vocab_embedding(vocab, vectors, embedding_size)
    # np.save("./data/{0}/vocab_embedding_{1}.npy".format(dataset, embedding_size), pretrained_word_embedding)
    
    if KB == "conceptnet":
        # Calculate edge matrix
        conceptnet = load_pickle("./data/KB/{0}.pkl".format(dataset))
        filtered_conceptnet = filter_conceptnet(conceptnet, vocab)
        filtered_conceptnet = remove_KB_duplicates(filtered_conceptnet)
        vocab_size = len(vocab.word2id)
        edge_matrix = np.zeros((vocab_size, vocab_size))
        for k in filtered_conceptnet:
            for c,w in filtered_conceptnet[k]:
                edge_matrix[vocab.word2id[k], vocab.word2id[c]] = w
        
        # reduce size of KB
        if KB_percentage > 0:
            logging.info("Keeping {0}% KB concepts...".format(KB_percentage*100))
            edge_matrix = edge_matrix * (np.random.random((vocab_size,vocab_size)) < KB_percentage).astype(float)
        edge_matrix = torch.FloatTensor(edge_matrix).to(device)
        edge_matrix[torch.arange(vocab_size), torch.arange(vocab_size)] = 1

        # incorporate NRC VAD intensity
        logging.info("Loading NRC...")
        NRC = load_pickle("./data/KB/NRC.pkl")
        affectiveness = np.zeros(vocab_size)
        for w, id in vocab.word2id.items():
            VAD = get_emotion_intensity(NRC, w)
            affectiveness[id] = VAD
        affectiveness = torch.FloatTensor(affectiveness).to(device)

    output_size = len(vocab.emotion2id)
    max_conversation_length_train = len(train[0])
    max_conversation_length_val = len(val[0])
    logging.info("Number of training utterances: {0}".format(vocab.num_utterances))
    logging.info("Average number of training utterances per conversation: {0}".format(vocab.num_utterances/len(train)))
    logging.info("Max conversation length in training set: {0}".format(max_conversation_length_train))
    logging.info("Max conversation length in validation set: {0}".format(max_conversation_length_val))
    logging.info("Emotion to ids: {0}".format(vocab.emotion2id))
    logging.info("Emotion distribution: {0}".format(vocab.emotion_freq_dist))
    train = convert_examples_to_ids(train, vocab)
    val = convert_examples_to_ids(val, vocab)

    train = flatten_examples_classification(train, vocab, k=context_length)
    val = flatten_examples_classification(val, vocab, k=context_length)
    logging.info("Batch size: {0}".format(batch_size))

    # model 
    logging.info("Building model...")
    model_kwargs = {
        "src_vocab": vocab_size,
        "tgt_vocab": vocab_size,
        "N": n_layers,
        "d_model": d_model,
        "d_ff": d_ff,
        "h": h,
        "output_size": output_size,
        "dropout": dropout,
        "KB": bool(KB),
        "model_variant": model_variant,
        "context_length": context_length,
        "graph_attention_variant": graph_attention_variant
    }

    model = make_model(**model_kwargs)
    
    # model initialization
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    if KB == "conceptnet":
        if GAW < 0:
            GAW = None
        model.graph_attention.init_params(GAW, edge_matrix, affectiveness, concentration_factor)
        
    logging.info("Initializing pretrained word embeddings into transformer...")
    model.src_embed[0].embedding.weight.data.copy_(torch.from_numpy(pretrained_word_embedding))
    model.tgt_embed[0].embedding.weight.data.copy_(torch.from_numpy(pretrained_word_embedding))
    if KB != "":
        model.graph_attention.concept_embed.weight.data.copy_(torch.from_numpy(pretrained_word_embedding))
    logging.info(model)
    logging.info("Number of model params: {0}".format(count_parameters(model)))
    model.to(device)

    # weighted crossentropy loss
    logging.info("Computing label weights...")
    label_weight = np.array(label_distribution_transformer(val))/np.array(label_distribution_transformer(train))
    label_weight = torch.tensor(label_weight/label_weight.sum()).float().to(device)*output_size
    logging.info("Label weight: {0}".format(label_weight))
    criterion = nn.CrossEntropyLoss(weight=label_weight, reduction="sum")
    optimizer = optim.Adam(model.parameters(), lr)

    # training
    train_epoch_losses = []
    val_epoch_losses = []
    logging.info("Start training...")
    for epoch in range(1, epochs + 1):
        train_batches = create_batches_classification(train, batch_size, vocab, train=True)
        val_batches = create_batches_classification(val, batch_size, vocab, train=False)
        
        train_epoch_loss = []
        val_epoch_loss = []
        model.train()
        loss_compute = SimpleLossCompute(model.generator, criterion, dataset, vocab.emotion2id, opt=optimizer, test=test_mode)

        for batch in train_batches:
            batch.to(device)
            out = model.forward(batch.src, batch.tgt, 
                                batch.src_mask, batch.tgt_mask)
            loss = loss_compute(out, batch.tgt_y, batch.ntokens)
            train_epoch_loss.append((loss/batch.ntokens).item())
        logging.info("-"*80)
        logging.info("Epoch {0}/{1}".format(epoch, epochs))
        logging.info("Training loss: {0:.4f}".format(np.mean(train_epoch_loss)))
        train_epoch_losses.append(np.mean(train_epoch_loss))
        score = loss_compute.score()
        loss_compute.clear()

        # validation
        # get src_attn
        src_attns = []
        model.eval()
        loss_compute = SimpleLossCompute(model.generator, criterion, dataset, vocab.emotion2id, opt=None, test=test_mode)
        with torch.no_grad():
            for batch in val_batches:
                batch.to(device)
                out = model.forward(batch.src, batch.tgt, 
                                    batch.src_mask, batch.tgt_mask)
                # get src attn
                src_attns.append(model.decoder.layers[0].src_attn.attn)
                loss = loss_compute(out, batch.tgt_y, batch.ntokens)
                val_epoch_loss.append((loss/batch.ntokens).item())
            logging.info("Validation loss: {0:.4f}".format(np.mean(val_epoch_loss)))
            val_epoch_losses.append(np.mean(val_epoch_loss))

        # get validation metrics
        score = loss_compute.score()
        loss_compute.clear()
        # logging.info("Validation score: {0}".format(score))

    