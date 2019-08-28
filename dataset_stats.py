import argparse
from collections import Counter
from utils.io import load_pickle
from utils.tools import counter_to_distribution

# datasets = ['EC','DD','MELD', "EmoryNLP","IEMOCAP"]
splits = ["train", "val", "test"]

def load_examples(dataset, split):
    data_path = "./data/{0}/{1}.pkl".format(dataset, split)
    return load_pickle(data_path)
    

def print_stats(examples):
    """
        examples: a list of examples, each example is a list of (utter, speaker, emotion, mask)
    """
    # stats
    utterances = []
    speakers = []
    emotions = []
    masks = []
    for ex in examples:
        for utter, speaker, emotion, mask in ex:
            if mask == 1:
                utterances.append(utter)
                speakers.append(speaker)
                emotions.append(emotion)
                masks.append(mask)
    print("-"*60)
    print("vocab size: ", len(set([w for utter in utterances for w in utter])))
    print("number of conversations: ", len(examples))
    print("number of utterances: ", len(utterances))
    print("average number of turns: {0:.2f}".format(len(utterances)/len(examples)))
    print("number of speakers: ", len(set(speakers)))
    print("speaker distribution: ", counter_to_distribution(Counter(speakers)))
    if len(set(emotions)) < 20:
        print("number of emotions: ", len(set(emotions)))
        print("emotion distribution: ", counter_to_distribution(Counter(emotions)))
    print("-"*60)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model for Context-based Emotion Classification in Conversations")
    parser.add_argument('--dataset', help='dataset to show stats', required=True, choices=['all', 'EC','DD','MELD', "EmoryNLP", "IEMOCAP"])
    args = parser.parse_args()
    if args.dataset == "all":
        datasets = ['EC','DD','MELD', "EmoryNLP", "IEMOCAP"]
    else:
        datasets = [args.dataset]
    
    for dataset in datasets:
        print("*"*80)
        print(dataset)
        for split in splits:
            examples = load_examples(dataset, split)
            print(split)
            print_stats(examples)
        print("*"*80)
        print("")

        
