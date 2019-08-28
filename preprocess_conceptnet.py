import csv
import argparse
from ast import literal_eval
from collections import defaultdict
from utils.io import load_pickle, to_pickle


def get_ngrams(utter, n):
        # utter: a list of tokens
        # n: up to n-grams
        total = []
        for i in range(len(utter)):
            for j in range(i, max(i-n, -1), -1):
                total.append("_".join(utter[j:i+1]))
        return total

# get all ngrams for a dataset
def get_all_ngrams(examples, n):
    all_ngrams = []
    for ex in examples:
        for utter, _,_,_ in ex:
            all_ngrams.extend(get_ngrams(utter, n))
    return set(all_ngrams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--n', default=1)
    args = parser.parse_args()

    dataset = args.dataset
    n = args.n
    
    print("Loading dataset...")
    train = load_pickle("./data/{0}/{1}.pkl".format(dataset, "train"))
    val = load_pickle("./data/{0}/{1}.pkl".format(dataset, "val"))
    test = load_pickle("./data/{0}/{1}.pkl".format(dataset, "test"))

    ngrams = get_all_ngrams(train+val+test, n)

    # get concepts for each ngram
    print("Loading conceptnet...")
    csv_reader = csv.reader(open("./data/KB/conceptnet-assertions-5.6.0.csv", "r"), delimiter="\t")
    concept_dict = defaultdict(set)

    for i, row in enumerate(csv_reader):
        if i%1000000 == 0:
            print("Processed {0} rows".format(i))
        
        lang = row[2].split("/")[2]
        if lang == 'en':
            c1 = row[2].split("/")[3]
            c2 = row[3].split("/")[3]
            weight = literal_eval(row[-1])["weight"]
            if c1 in ngrams:
                concept_dict[c1].add((c2, weight))
            if c2 in ngrams:
                concept_dict[c2].add((c1, weight))
    print("Saving concepts...")
    to_pickle(concept_dict, "./data/KB/{0}.pkl".format(dataset))