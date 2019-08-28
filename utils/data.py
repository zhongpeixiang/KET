import re
import csv
import random
from collections import Counter, defaultdict

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import spacy
from spacy.lang.en.stop_words import STOP_WORDS as spacy_stopwords


DD_label_to_emotions = {0: "neutral", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}
IEMOCAP_emotions_to_label = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
IEMOCAP_label_to_emotions = {0: 'hap', 1: 'sad', 2: 'neu', 3: 'ang', 4: 'exc', 5: 'fru'}
nltk_stopwords = stopwords.words('english')
# spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS # older version of spacy
stopwords = set(nltk_stopwords).union(spacy_stopwords)
porter = PorterStemmer()

class Vocab(object):
    def __init__(self, examples, min_freq, max_vocab_size):
        """
            examples: a list of examples, each example is a list of (utter, speaker, emotion, mask), followed by an emotion label
            min_freq: the min frequency of word in the vocabulary
            max_vocab_size: max vocabulary size

            return: a Vocab object
        """
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size
        
        self.word2id = {"<pad>": 0, "<unk>": 1}
        self.id2word = {0: "<pad>", 1: "<unk>"}
        self.speaker2id = {}
        self.emotion2id = {}

        self.word_freq_dist = Counter()
        self.speaker_freq_dist = Counter()
        self.emotion_freq_dist = Counter()
        
        utterance_lengths = []
        conversation_lengths = []

        for ex in examples:
            conv_length = sum([utter[-1] for utter in ex])
            conversation_lengths.append(conv_length)
            for utter, speaker, emotion, mask in ex:
                if mask == 1:
                    self.word_freq_dist.update(utter)
                    self.speaker_freq_dist.update([speaker])
                    self.emotion_freq_dist.update([emotion])
                    utterance_lengths.append(len(utter))

        # filter by min_freq
        words = [(w,cnt) for w, cnt in self.word_freq_dist.items() if cnt >= min_freq]
        words = sorted(words, key=lambda x: x[1], reverse=True)

        # cap vocab size
        words = words[:max_vocab_size]

        # build word2id and id2word
        for idx, (w, cnt) in enumerate(words):
            self.word2id[w] = idx+2
            self.id2word[idx+2] = w

        # build speaker to id
        speakers = sorted(self.speaker_freq_dist.items(), key=lambda x: x[1], reverse=True)
        for idx, (speaker, cnt) in enumerate(speakers):
            self.speaker2id[speaker] = idx

        # build emotion to id
        emotions = sorted(self.emotion_freq_dist.items(), key=lambda x: x[1], reverse=True)
        for idx, (emotion, cnt) in enumerate(emotions):
            self.emotion2id[emotion] = idx
        
        self.max_conversation_length = max(conversation_lengths)
        self.max_sequence_length = max(utterance_lengths)
        self.num_utterances = len(utterance_lengths)
        

def convert_examples_to_ids(examples, vocab):
    """
        examples: a list of examples, each example is a list of (utter, speaker, emotion, mask), followed by an emotion label
        vocab: a Vocab object
        max_sequence_length: max sequence length
        
        return: examples containing word ids
    """
    examples_ids = []
    for ex in examples:
        ex_ids = []
        for utter, speaker, emotion, mask in ex:
            utter_ids = [vocab.word2id[w] if w in vocab.word2id else vocab.word2id["<unk>"] for w in utter]
            utter_ids = utter_ids + (vocab.max_sequence_length - len(utter_ids)) * [vocab.word2id["<pad>"]]
            if speaker in vocab.speaker2id:
                speaker_id = vocab.speaker2id[speaker]
            else:
                speaker_id = len(vocab.speaker2id) # val or test set, use a new speaker id
            emotion_id = vocab.emotion2id[emotion]
            ex_ids.append((utter_ids, speaker_id, emotion_id, mask))
        examples_ids.append(ex_ids)
    return examples_ids

def create_one_batch(examples):
    """
        examples: a batch of examples having the same number of turns and seq_len, each example is a list of (utter, speaker, label, mask)
        
        return: batch_x, batch_y, where batch_x is a tuple of token ids and speaker ids
    """
    tokens = []
    speakers = []
    labels = []
    masks = []
    for ex in examples:
        ex_tokens, ex_speakers, ex_labels, ex_masks = list(zip(*ex))
        tokens.append(ex_tokens)
        speakers.append(ex_speakers)
        labels.append(ex_labels)
        masks.append(ex_masks)
    
    return (tokens, speakers), labels, masks


def create_batches(examples, batch_size, train=True):
    batch_data = []
    if train == True:
        random.shuffle(examples)
    
    batch_ids = list(range(0, len(examples), batch_size)) + [len(examples)]
    for s, e in zip(batch_ids[:-1], batch_ids[1:]):
        batch_examples = examples[s:e]
        batch_x, batch_y, batch_mask = create_one_batch(batch_examples)
        batch_data.append((batch_x, batch_y, batch_mask))
    return batch_data


def create_balanced_batches(examples, batch_size, train=True):
    batch_data = []
    if train == True:
        random.shuffle(examples)
    
    num_valid_utterances = []
    for ex in examples:
        num_valid_utterances.append(sum([mask for utter, speaker, label, mask in ex]))
    
    batch_ids = [0]
    for i in range(len(examples)):
        if len(num_valid_utterances[batch_ids[-1]:i]) >= batch_size:
            batch_ids.append(i)
    
    for s, e in zip(batch_ids[:-1], batch_ids[1:]):
        batch_examples = examples[s:e]
        batch_x, batch_y, batch_mask = create_one_batch(batch_examples)
        batch_data.append((batch_x, batch_y, batch_mask))
    return batch_data


def merge_splits(train, val):
    if len(train[0]) > len(val[0]):
        num_additional_utterances = len(train[0]) - len(val[0])
        for ex in val:
            if ex[-1][0] == ['this', 'is', 'a', 'dummy', 'sentence']:
                dummy_ex = ex[-1]
                break
        new_val = []
        for ex in val:
            new_val.append(ex + num_additional_utterances*[dummy_ex])
        return train + new_val
    elif len(train[0]) < len(val[0]):
        num_additional_utterances = len(val[0]) - len(train[0])
        for ex in train:
            if ex[-1][0] == ['this', 'is', 'a', 'dummy', 'sentence']:
                dummy_ex = ex[-1]
                break
        new_train = []
        for ex in train:
            new_train.append(ex + num_additional_utterances*[dummy_ex])
        return new_train + val
    else:
        return train + val

# conceptnet
def get_vocab_embedding(vocab, vectors, embedding_size):
    pretrained_word_embedding = np.zeros((len(vocab.word2id), embedding_size))
    for w, i in vocab.word2id.items():
        pretrained_word_embedding[i] = vectors.query(w)
    return pretrained_word_embedding


def get_emotion_intensity(NRC, word):
    if word not in NRC:
        word = porter.stem(word)
        if word not in NRC:
            return 0.5
    v, a, d = NRC[word]
    a = a/2
    return (np.linalg.norm(np.array([v, a]) - np.array([0.5, 0])) - 0.06467)/0.607468


# edge matrix construction
def filter_conceptnet(conceptnet, vocab):
    filtered_conceptnet = {}
    for k in conceptnet:
        if k in vocab.word2id and k not in stopwords:
            filtered_conceptnet[k] = set()
            for c,w in conceptnet[k]:
                if c in vocab.word2id and c not in stopwords and w>=1:
                    filtered_conceptnet[k].add((c,w))
    return filtered_conceptnet


# remove cases where the same concept has multiple weights
def remove_KB_duplicates(conceptnet):
    filtered_conceptnet = {}
    for k in conceptnet:
        filtered_conceptnet[k] = set()
        concepts = set()
        filtered_concepts = sorted(conceptnet[k], key=lambda x: x[1], reverse=True)
        for c,w in filtered_concepts:
            if c not in concepts:
                filtered_conceptnet[k].add((c, w))
                concepts.add(c)
    return filtered_conceptnet
