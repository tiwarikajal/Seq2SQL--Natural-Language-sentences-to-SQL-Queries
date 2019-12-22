from __future__ import unicode_literals, print_function, division

from collections import defaultdict
from io import open
import unicodedata
import torch
from util.constants import *
import nltk
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LanguageUtil:
    def __init__(self, name):
        self.name = name
        self.word2index = defaultdict(int)
        self.word2count = defaultdict(int)
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence:
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLangs(lang1, lang2):
    print("Reading lines...")

    lines = pd.read_json("data/tokenized_train.jsonl", lines=True)

    # Split every line into pairs and normalize
    pairs = []
    for idx, row in lines.iterrows():
        tokens_en = row["tokenized_question"]
        tokens_sql = row["tokenized_query"]
        pairs.append([tokens_en, tokens_sql])

    input_lang = LanguageUtil(lang1)
    output_lang = LanguageUtil(lang2)

    return input_lang, output_lang, pairs


def prepareValData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs_val(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def readLangs_val(lang1, lang2):
    print("Reading lines...")

    lines = pd.read_json("data/tokenized_dev.jsonl", lines=True)

    # Split every line into pairs and normalize
    pairs = []
    for idx, row in lines.iterrows():
        tokens_en = row["tokenized_question"]
        tokens_sql = row["tokenized_query"]
        pairs.append([tokens_en, tokens_sql])

    input_lang = LanguageUtil(lang1)
    output_lang = LanguageUtil(lang2)

    return input_lang, output_lang, pairs


def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print("Read %s sentence pairs" % len(pairs))
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence]


def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)
