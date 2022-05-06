#!/usr/bin/env python3
# https://pytorch.org/tutorials/beginner/chatbot_tutorial.html

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv, random, re, os, math
import unicodedata, codecs
import itertools


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

corpus_name = "corpus movie-dialogs corpus"
corpus_dir = "data"
corpus = os.path.join(corpus_dir, corpus_name)

def print_lines(file, n=10):
    with open(file, "rb") as datafile:
        lines = datafile.readlines()
    for line in lines[:n]:
        print(line)

# splits each line of the file into a dictionary of fields
def load_lines(file_name, fields):
    lines = {}
    with open(file_name, 'r', encoding="iso-8859-1") as f:
        for line in f:
            values = line.split(" +++$+++ ")
            line_obj = {}
            for i, field in enumerate(fields):
                line_obj[field] = values[i]
            lines[line_obj["lineID"]] = line_obj
    return lines

# group fields of lines from `load_lines` into conversations based on `movie_conversations.txt`
def load_conversations(file_name, lines, fields):
    conversations = []
    with open(file_name, 'r', encoding="iso-8859-1") as f:
        for line in f:
            values = line.split(" +++$+++ ")
            conv_obj = {}
            for i, field in enumerate(fields):
                conv_obj[field] = values[i]
            # convert string to list (conv_obj["utteranceIDs"] == "['L598485', 'L598486', ...]")
            utterance_id_pattern = re.compile("L[0-9]+")
            line_ids = utterance_id_pattern.findall(conv_obj["utteranceIDs"])
            conv_obj["lines"] = []
            for line_id in line_ids:
                conv_obj["lines"].append(lines[line_id])
            conversations.append(conv_obj)
    return conversations

# extract pairs of sentence from conversations
def extract_sentence_pairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1):
            input_line = conversation["lines"][i]["text"].strip()
            target_line = conversation["lines"][i+1]["text"].strip()
            if input_line and target_line:
                qa_pairs.append([input_line, target_line])
    return qa_pairs


datafile = os.path.join(corpus_dir, "formatted_movie_lines.txt")

delimiter = "\t"
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

lines = {}
conversation = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["charater1ID", "character2ID", "movieID", "utteranceIDs"]

print("Processing corpus...")
lines = load_lines(os.path.join(corpus_dir, "movie_lines.txt"), MOVIE_LINES_FIELDS)
print("Loading conversation...")
conversations = load_conversations(os.path.join(corpus_dir, "movie_conversations.txt"), lines, MOVIE_CONVERSATIONS_FIELDS)

print("Writing newly formatted file...")
with open(datafile, 'w', encoding="utf-8") as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator="\n")
    for pair in extract_sentence_pairs(conversations):
        writer.writerow(pair)

print("Sample lines from file:")
print_lines(datafile)

# used for padding short sentences
PAD_token = 0
# start-of-sentence token
SOS_token = 1 
# end-of-sentence
EOS_token = 2

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        # count SOS, EOS, PAD
        self.num_words = 3

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:
            self.word2count[word] += 1

    # remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed:
            return
        self.trimmed = True

        keep_words = []
        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print("keep_words {} / {} = {:.4f}".format(len(keep_words), 
            len(self.word2index), len(keep_words) / len(self.word2index)))

        # reinitialize dictionaries
        self. word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
        self.num_words = 3

        for word in keep_words:
            self.add_word(word)


# maximum sentence length to consider
MAX_LENGTH = 10

# turn Unicode to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
            c for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
            )

# lowercase, trim, and remove non-letter character
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub("[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

# read query/response pairs and return a voc object
def read_vocs(datafile, corpus_name):
    print("Reading lines...")
    # read the file and split into lines
    lines = open(datafile, encoding="utf-8").read().strip().split("\n")
    pairs = [[normalize_string(s) for s in l.split("\t")] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs

# return true if both sentences in a pair 'p' are under the MAX_LENGTH threshold
def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

# return a populated voc object and pair list
def load_prepared_data(corpus, corpus_name, datafile, save_dir):
    print("Start preparing training data...")
    voc, pairs = read_vocs(datafile, corpus_name)
    print("Read {!s} sentence pairs".format(len(pairs)))
    pairs = filter_pairs(pairs)
    print("Trimmed to {!s} sentence pairs".format(len(pairs)))
    print("Counting words...")
    for pair in pairs:
        voc.add_sentence(pair[0])
        voc.add_sentence(pair[1])
    print("Counted words: ", voc.num_words)
    return voc, pairs


save_dir = os.path.join("dats", "save")
voc, pairs = load_prepared_data(corpus, corpus_name, datafile, save_dir)
print("pairs:")
for pair in pairs[:10]:
    print(pair)

# minimum word count threshold for trimming
MIN_COUNT = 3
