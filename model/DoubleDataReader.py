import numpy as np
import torch
from torch.utils.data import Dataset
from FILEPATHS import *


class DoubleDataReader:
    NEGATIVE_TABLE_SIZE = 1600

    def __init__(self, inputFileName, min_count):

        self.negatives = []
        self.discards = []
        self.negpos = 0
        
        self.word2id = dict()
        self.id2word = dict()
        self.sentences_count = 0
        self.token_count = 0
        self.word_frequency = dict()
        self.word_neighbor = dict()

        self.inputFileName = inputFileName
        self.read_words(min_count)
        self.initTableNegatives()
        self.initTableDiscards()

    def read_words(self, min_count):
        word_frequency = dict()
        word_neighbor = { i:[] for i in range(n_points)}
        # print(word_neighbor)
        for line in open(self.inputFileName, encoding="utf8"):
            line = line.split()
            # print("line", line)
            if len(line) > 1:
                self.sentences_count += 1
                for i, word in enumerate(line):
                    if i == 2: continue
                    if i == 0: 
                        word_neighbor[int(line[0])].append(int(line[1]))

                    self.token_count += 1
                    word_frequency[int(word)] = word_frequency.get(int(word), 0) + 1

        
        self.word_neighbor = word_neighbor
        wid = 0
        for w, c in word_frequency.items():        
            self.word2id[w] = w
            self.id2word[w] = w 
            self.word_frequency[w] = c            
            wid += 1
        print("Total embeddings: " + str(len(self.word_neighbor)))

    def initTableNegatives(self):
        target_entities = {w:c for w, c in self.word_frequency.items() if n_courses <= int(w) }
        pow_frequency = np.array(list(target_entities.values())) ** 0.5
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * DoubleDataReader.NEGATIVE_TABLE_SIZE)
        for wid, c in enumerate(count): 
            self.negatives += [wid] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    def getNegatives(self, target, size):  
        negs = []
        while len(negs) < size:
            shortage = size - len(negs)
            pos_s, pos_e = self.negpos, min(self.negpos + shortage, len(self.negatives))
            samples = [u for u in self.negatives[pos_s:pos_e] if not target in self.word_neighbor[u]]
            self.negpos = pos_e % len(self.negatives)
            negs = np.concatenate((samples, negs))
        return negs

    def initTableDiscards(self):
        t = 0.0001
        f = np.array(list(self.word_frequency.values())) / self.token_count
        self.discards = np.sqrt(t / f) + (t / f)


# ----------------------------------------------------------------------------------------------------------------- #


class DoubleCrossDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                u, v, w = line.split()
                if len(self.data.getNegatives(int(v), 5)) == 0: print("zero_negative", int(u), int(v))
                return [(int(u), int(v), self.data.getNegatives(int(v), 5), float(w))]
                
    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _, _ in batch] 
        all_v = [v for batch in batches for _, v, _ , _ in batch] 
        all_neg_v = [neg_v for batch in batches for _, _, neg_v, _ in batch] 
        all_w = [w for batch in batches for _, _, _, w in batch] 
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v), torch.FloatTensor(all_w) 


class DoubleSimilarDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.input_file = open(data.inputFileName, encoding="utf8")

    def __len__(self):
        return self.data.sentences_count

    def __getitem__(self, idx):
        while True:
            line = self.input_file.readline()
            if not line:
                self.input_file.seek(0, 0)
                line = self.input_file.readline()

            if len(line) > 1:
                u, v, s = line.split()
                return [(int(u), int(v), float(s))]
                
    @staticmethod
    def collate(batches):
        all_u = [u for batch in batches for u, _, _, _ in batch] 
        all_v = [v for batch in batches for _, v, _, _ in batch] 
        all_l = [l for batch in batches for _, _, l, _ in batch] 
        all_s = [s for batch in batches for _, _, _, s in batch] 
        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.FloatTensor(all_l), torch.FloatTensor(all_s) 
