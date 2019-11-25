import spacy
import argparse
from tqdm import tqdm
import pandas as pd
import pickle
import json
from functools import partial
from collections import Counter

from utils.tools import counter_to_distribution

spacy_en = spacy.load("en")


def create_examples_EC(split):
    """
        split: train, val or test
        return: a list of examples, each example is a list of utterances and an emotion label for the last utterance
    """
    with open("./data/EC/{0}.txt".format(split), "r") as f:
        f.readline()
        conversations = f.readlines()
    print("{0} split has {1} conversations".format(split, len(conversations)))
    print("max_conv_length: ", 3)
    
    examples = []
    dummy_emotion = conversations[0].strip().split("\t")[-1]
    for conv in conversations:
        utterances_emotion = [e.strip() for e in conv.split("\t")][1:]
        ex = []
        for idx, utter in enumerate(utterances_emotion[:-1]):
            if idx%2 == 0:
                speaker = "Speaker A"
            else:
                speaker = "Speaker B"
            if idx <= 1:
                ex.append((utter, speaker, dummy_emotion, 0))
            elif idx == 2:
                ex.append((utter, speaker, utterances_emotion[-1], 1))
        examples.append(ex)
    return examples


def create_examples_DD(split):
    """
        split: train, val or test
        return: a list of examples, each example is a list of utterances and an emotion label for the last utterance
    """
    with open("./data/DD/dialogues_{0}.txt".format(split), "r") as f:
        conversations = f.readlines()
    with open("./data/DD/dialogues_emotion_{0}.txt".format(split), "r") as f:
        emotions = f.readlines()
    print("{0} split has {1} conversations".format(split, len(conversations)))
    
    examples = []
    max_conv_length = max([len(conv.split("__eou__")[:-1]) for conv in conversations])
    print("max_conv_length: ", max_conv_length)
    
    dummy_utterance = "this is a dummy sentence"
    dummy_emotion = emotions[0].strip().split(" ")[0]
    dummy_speaker = "Speaker A"
    for conv, emo in zip(conversations, emotions):
        utterances = [utter.strip() for utter in conv.split("__eou__")][:-1]
        
        # add speaker info
        speakers = []
        for idx, utter in enumerate(utterances):
            if idx%2 == 0:
                speaker = "Speaker A"
            else:
                speaker = "Speaker B"
            speakers.append(speaker)
        
        # add emotion
        utter_emotions = [emotion.strip() for emotion in emo.split(" ")[:-1]]
        assert len(utterances) == len(speakers) == len(utter_emotions)
                
        # pad dummy utterances
        conv_length = len(utterances)
        masks = conv_length*[1] + (max_conv_length-conv_length)*[0]
        utterances += (max_conv_length-conv_length)*[dummy_utterance]
        utter_emotions += (max_conv_length-conv_length)*[dummy_emotion]
        speakers += (max_conv_length-conv_length)*[dummy_speaker]
        
        # create examples
        examples.append(list(zip(utterances, speakers, utter_emotions, masks)))
    return examples


def create_examples_MELD(split):
    """
        split: train, val or test
        return: a list of examples, each example is a list of utterances and an emotion label for the last utterance
    """
    data = pd.read_csv("./data/MELD/{0}.csv".format(split))
    print("{0} split has {1} conversations".format(split, data["Dialogue_ID"].unique().shape[0]))
    
    examples = []
    conv_lengths = []
    for idx, conv in data.groupby("Dialogue_ID"):
        conv_lengths.append(len(conv))
    max_conv_length = max(conv_lengths)
    print("max_conv_length: ", max_conv_length)
    
    dummy_utterance = "this is a dummy sentence"
    dummy_emotion = data["Emotion"][0]
    dummy_speaker = data["Speaker"][0]
    for idx, conv in data.groupby("Dialogue_ID"):
        utterances = conv["Utterance"].tolist()
        speakers = conv["Speaker"].tolist()
        emotions = conv["Emotion"].tolist()
        
        assert len(utterances) == len(speakers) == len(emotions)
                
        # pad dummy utterances
        conv_length = len(utterances)
        masks = conv_length*[1] + (max_conv_length-conv_length)*[0]
        utterances += (max_conv_length-conv_length)*[dummy_utterance]
        emotions += (max_conv_length-conv_length)*[dummy_emotion]
        speakers += (max_conv_length-conv_length)*[dummy_speaker]
        
        # associate utterances with speakers
        utterances = list(zip(utterances, speakers, emotions, masks))

        examples.append(utterances)
    return examples


def create_examples_EmoryNLP(split):
    """
        split: train, val or test
        return: a list of examples, each example is a list of utterances and an emotion label for the last utterance
    """
    data = pd.read_csv("./data/EmoryNLP/{0}.csv".format(split))
    print("{0} split has {1} conversations".format(split, data[["Season", "Episode", "Scene_ID"]].drop_duplicates().shape[0]))
    
    examples = []
    conv_lengths = []
    for idx, conv in data.groupby(["Season", "Episode", "Scene_ID"]):
        conv_lengths.append(len(conv))
    max_conv_length = max(conv_lengths)
    print("max_conv_length: ", max_conv_length)
    
    dummy_utterance = "this is a dummy sentence"
    dummy_emotion = data["Emotion"][0]
    dummy_speaker = data["Speaker"][0][2:-2]
    for idx, conv in data.groupby(["Season", "Episode", "Scene_ID"]):
        utterances = conv["Utterance"].tolist()
        speakers = [speaker[2:-2] for speaker in conv["Speaker"].tolist()]
        emotions = conv["Emotion"].tolist()
        
        assert len(utterances) == len(speakers) == len(emotions)
                
        # pad dummy utterances
        conv_length = len(utterances)
        masks = conv_length*[1] + (max_conv_length-conv_length)*[0]
        utterances += (max_conv_length-conv_length)*[dummy_utterance]
        emotions += (max_conv_length-conv_length)*[dummy_emotion]
        speakers += (max_conv_length-conv_length)*[dummy_speaker]
        
        # associate utterances with speakers
        utterances = list(zip(utterances, speakers, emotions, masks))

        examples.append(utterances)
    return examples

"""
# Regarding producing data.pkl for IEMOCAP:
# The exact code for preprocessing IEMOCAP raw to data.pkl is left somewhere in my older server
# The following code is not gurrenteed to work and just intended to give some clues.

dataset = pickle.load(open("./data/IEMOCAP/IEMOCAP_features_raw.pkl", 'rb'), encoding='latin1')
# randomly select validation sessions
train_ids = list(dataset[7])
val_ids = []
num_vals = 20

while len(set(val_ids)) < num_vals:
    random_id = random.choice(train_ids)
    val_ids.append(random_id)
val_ids = list(set(val_ids))
train_ids = list(set(train_ids) - set(val_ids))
test_ids = list(dataset[8])

new_dataset = []
new_dataset.append(dataset[1]) # speaker
new_dataset.append(dataset[6]) # utterances
new_dataset.append(dataset[2]) # label
new_dataset.append(train_ids) 
new_dataset.append(val_ids)
new_dataset.append(test_ids)
    
# save new dataset
with open("./data/IEMOCAP/data.pkl", "wb") as f:
pickle.dump(new_dataset, f)
"""

def create_examples_IEMOCAP(split):
    dataset = pickle.load(open("./data/IEMOCAP/data.pkl", 'rb'))
    if split == "train":
        session_ids = dataset[3]
    elif split == "val":
        session_ids = dataset[4]
    else:
        session_ids = dataset[5]
    print("{0} split has {1} conversations".format(split, len(session_ids)))
        
    examples = []
    conv_lengths = []
    for i in session_ids:
        conv_lengths.append(len(dataset[1][i]))
        dummy_emotion = dataset[2][i][0]
        dummy_speaker = dataset[0][i][0]
    max_conv_length = max(conv_lengths)
    print("max_conv_length: ", max_conv_length)
    
    dummy_utterance = "this is a dummy sentence"
    for i in session_ids:
        speakers = dataset[0][i]
        utterances = dataset[1][i]
        emotions = dataset[2][i]
        assert len(speakers) == len(utterances) == len(emotions)
        
        # pad dummy utterances
        conv_length = len(utterances)
        masks = conv_length*[1] + (max_conv_length-conv_length)*[0]
        utterances += (max_conv_length-conv_length)*[dummy_utterance]
        emotions += (max_conv_length-conv_length)*[dummy_emotion]
        speakers += (max_conv_length-conv_length)*[dummy_speaker]
        
        # associate utterances with speakers
        utterances = list(zip(utterances, speakers, emotions, masks))

        examples.append(utterances)
            
    return examples


def clip_conversation_length(examples, max_conversation_length):
    """
        examples: a list of examples
        max_conversation_length: the max number of utterances in one example
        return: a list of clipped examples where each example is limited to the most recent k utterances
    """
    clipped_examples = []
    num_clips = 0
    for ex in examples:
        if len(ex) > max_conversation_length+1:
            num_clips += 1
            ex = ex[-(max_conversation_length+1):]
        clipped_examples.append(ex)
    print("Number of clipped examples: {0}".format(num_clips))
    return clipped_examples


def clean(text, max_sequence_length):
    """
        text: a piece of text in str
        max_sequence_length: the max sequence length for each utterance
        return: a list tokenized cleaned words
    """
    # lower case
    text = text.lower()
    
    # other cleaning processes
    
    # tokenization
    return [token.text for token in spacy_en.tokenizer(text)][:max_sequence_length]


def clean_examples(examples, max_sequence_length):
    """
        examples: a list of examples, each example is a list of (utterance, speaker, emotion, mask)
        max_sequence_length: the max sequence length for each utterance
        return: a list tokenized cleaned examples
    """
    cleaned_examples = []
    for ex in tqdm(examples):
        cleaned_examples.append([(clean(utterance, max_sequence_length), speaker, emotion, mask) 
                                 for utterance, speaker, emotion, mask in ex])
    return cleaned_examples


create_examples_dict = {
    "EC": create_examples_EC,
    "DD": create_examples_DD,
    "MELD": create_examples_MELD,
    "EmoryNLP": create_examples_EmoryNLP,
    "IEMOCAP": create_examples_IEMOCAP
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data preprocessing script")
    parser.add_argument('--dataset', help='Dataset to preprocess', choices=['EC','DD','MELD', "EmoryNLP", "IEMOCAP"], required=True)
    parser.add_argument('--max_conversation_length', type=int, default=10)
    parser.add_argument('--max_sequence_length', type=int, default=30)
    
    # parse args
    args = parser.parse_args()
    dataset = args.dataset
    max_conversation_length = args.max_conversation_length
    max_sequence_length = args.max_sequence_length
    
    # create examples
    print("Preprocessing {0}...".format(dataset))
    create_examples = create_examples_dict[dataset]
    for split in ["train", "val", "test"]:
        examples = create_examples(split)
        # examples = clip_conversation_length(examples, max_conversation_length)
        examples = clean_examples(examples, max_sequence_length)
        
        # save data
        path_to_save = "./data/{0}/{1}.pkl".format(dataset, split)
        print("Saving data to {0}".format(path_to_save))
        with open(path_to_save, "wb") as f:
            pickle.dump(examples, f)
