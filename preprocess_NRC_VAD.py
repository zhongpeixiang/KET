import pickle
import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("./data/KB/NRC-VAD-Lexicon.txt", sep='\t')

    NRC = {}
    for i, row in df.iterrows():
        NRC[row[0]] = tuple(row[1:])

    pickle.dump(NRC,"./data/KB/NRC.pkl")