import numpy as np
import torch

# no tgt mask for classification
def no_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size) # size: seq_len
    subsequent_mask = np.zeros(attn_shape).astype('uint8')
    # subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    # print("subsequent mask shape: ", subsequent_mask.shape) # (1, seq_len-1, seq_len-1)
    return torch.from_numpy(subsequent_mask) == 0


class ClassificationBatch:
    "Object for holding a batch of data with mask during training"
    def __init__(self, src, tgt, label, pad=0, concept=None):
        # print("src, tgt shape:", src.shape, tgt.shape)
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        self.tgt = tgt
        self.concept = concept
        self.concept_mask = None
        if concept is not None:
            self.concept_mask = (concept != pad).unsqueeze(-2)
        self.tgt_mask = (tgt != pad).unsqueeze(-2)
        self.tgt_y = label
        # self.tgt_mask = self.make_std_mask(self.tgt, pad)
        self.ntokens = torch.FloatTensor([len(src)])[0] # batch_size
        
    def to(self, device):
        self.src = self.src.to(device)
        self.src_mask = self.src_mask.to(device)
        self.tgt = self.tgt.to(device)
        self.tgt_mask = self.tgt_mask.to(device)
        if self.concept is not None:
            self.concept = self.concept.to(device)
            self.concept_mask = self.concept_mask.to(device)
        self.tgt_y = self.tgt_y.to(device)
        self.ntokens = self.ntokens.to(device)
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words"
        tgt_mask = (tgt != pad).unsqueeze(-2) # (batch_size, 1, seq_len-1)
        # print("tgt_mask shape: ", tgt_mask.shape)
        tgt_mask = tgt_mask & no_mask(tgt.size(-1)).type_as(tgt_mask)
        return tgt_mask


def flatten_examples_classification(examples, vocab, k=1):
    # returns a list of ([(utter1, speaker1), (utter2, speaker2), ..., (utterk, speakerk)], label)
    import random
    classfication_examples = []
    all_speakers = list(vocab.speaker2id.values())
    empty_ex = len(examples[0][0][0])*[vocab.word2id["<pad>"]]
    for ex in examples:
        for i in range(len(ex)):
            mask = ex[i][-1]
            if mask == 1:
                if i < k:
                    context = [(empty_ex.copy(), random.choice(all_speakers)) for i in range(k-i)] # k-i
                    context += [(ex[i-j][0].copy(), ex[i-j][1]) for j in range(i, 0, -1)] # (k-i) + (i) = k
                else:
                    context = [(ex[i-j][0].copy(), ex[i-j][1]) for j in range(k, 0, -1)] # k
                if k > 0:
                    new_ex = (context + [(ex[i][0].copy(), ex[i][1])], ex[i][2])
                else:
                    new_ex = [(empty_ex.copy(), random.choice(all_speakers)), (ex[i][0].copy(), ex[i][1])], ex[i][2] # add one empty context utterance
                # if i==0:
                #     random_speaker = random.choice(all_speakers)
                #     new_ex = [(empty_ex.copy(), random_speaker), (ex[i][0].copy(), ex[i][1])], ex[i][2]
                # else:
                #     new_ex = [(ex[i-1][0].copy(), ex[i-1][1]), (ex[i][0].copy(), ex[i][1])], ex[i][2]
                classfication_examples.append(new_ex)
    return classfication_examples


def create_batches_classification(examples, batch_size, vocab, train=True):
    import random
    
    def create_one_batch(examples, vocab):
        """
            examples: a batch of examples having the same number of turns and seq_len, 
                each example is a list of ([(utter1, speaker1), ..., (utterk, speakerk), (utterA, speakerA)], label)

            return: batch_Q, batch_Q_speakers, batch_A, batch_A_speakers, batch_label
        """
        Qs = [] # context utterances
        Q_speakers = [] # context speakers
        As = [] # current utterance
        A_speakers = [] # current speaker
        labels = [] # label of current utterance

        for ex in examples:
            context = []
            context_speakers = []
            for Q, Q_speaker in ex[0][:-1]:
                context.extend(Q)
                context_speakers.append(Q_speaker)
            A, A_speaker = ex[0][-1]
            label = ex[1]
            
            Qs.append(context)
            Q_speakers.append(context_speakers)
            As.append(A)
            A_speakers.append(A_speaker)
            labels.append(label)
        
        batch = ClassificationBatch(torch.LongTensor(Qs), torch.LongTensor(As), torch.LongTensor(labels), vocab.word2id["<pad>"])
        return batch

    batch_data = []
    if train == True:
        random.shuffle(examples)
    batch_ids = list(range(0, len(examples), batch_size)) + [len(examples)]
    for s, e in zip(batch_ids[:-1], batch_ids[1:]):
        batch_examples = examples[s:e]
        one_batch = create_one_batch(batch_examples, vocab)
        batch_data.append(one_batch)
    return batch_data
