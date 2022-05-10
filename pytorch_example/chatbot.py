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

def trim_rare_words(voc, pairs, MIN_COUNT):
    voc.trim(MIN_COUNT)
    # filter out pairs with trimmed words
    keep_pairs = []
    for pair in pairs:
        input_sentence = pair[0]
        output_sentence = pair[1]
        keep_input = True
        keep_output = True
        for word in input_sentence.split(' '):
            if word not in voc.word2index:
                keep_input = False
                break
        for word in output_sentence.split(' '):
            if word not in voc.word2index:
                keep_output = False
                break

        if keep_input and keep_output:
            keep_pairs.append(pair)

    print("Trimmed from {} pairs to {}, {:.4f} of total.".format(len(pairs), 
            len(keep_pairs), len(keep_pairs) / len(pairs)))
    return keep_pairs

# trim voc and pairs
pairs = trim_rare_words(voc, pairs, MIN_COUNT)

def indexes_from_sentence(voc, sentence):
    return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]

def zero_padding(l, fillvalue=PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binary_matrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

# returns padding input sequence tensor and lengths
def input_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, lengths

# return padded target sequence tensor, padding mask, and max target length
def output_var(l, voc):
    indexes_batch = [indexes_from_sentence(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    pad_list = zero_padding(indexes_batch)
    mask = binary_matrix(pad_list)
    mask = torch.BoolTensor(mask)
    pad_var = torch.LongTensor(pad_list)
    return pad_var, mask, max_target_len

# return all items for a given batch of pairs
def batch_to_train_data(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(' ')), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_var(input_batch, voc)
    output, mask, max_target_len = output_var(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

# example for validation
small_batch_size = 5
batches = batch_to_train_data(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

print("input_variable: ", input_variable)
print("lengths: ", lengths)
print("target_variable: ", target_variable)
print("mask: ", mask)
print("max_target_len: ", max_target_len)

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding

        # initialize GRU. the input_size and hidden_size params are both set to `hidden_size`
        # because out input size is a word embedding with number of features == `hidden_size`
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_length, hidden=None):
        # convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        # return outputs and final hidden state
        return outputs, hidden

# Luong attention layer
class Attn(nn.Module):
    def __init__(self, method, hidden):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # return the softmax normalized probability scores (with added dimensions)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers==1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # note: we run this one step (word) at a time
        # get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        return output, hidden

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()

def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, 
        decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=MAX_LENGTH):
    # zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # set device options
    input_variable = input_variable.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # length for rnn packing should always be on the cpu
    lengths = lengths.to("cpu")

    # initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # determine if we are using teacher forcing this iteration
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            # no teacher forcing: next input id decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[top[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    
    # backpropagation
    loss.backward()

    # clip gradients: gradients are modified in place 
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals

def train_iters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, 
        decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, 
        n_iteration, batch_size, print_every, save_every, clip, corpus_name, load_filename):
    # load batches for each iteration
    training_batches = [batch_to_train_data(voc, [random.choice(pairs) for _ in range(batch_size)]) 
            for _ in range(n_iteration)]
    # initializations
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batche = training_batches[iteration - 1]
        # extract field from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        # run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder, 
                decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss

        # print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}, Percent complete: {:.1f}%; Average loss: {:.4f}".format(
                    iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # save checkpoint
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}-{}'.format(
                encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists((directory):
                    os.makedirs((directory)
            torch.save({
                "iteration": iteration,
                "en": encoder.state_dict(),
                "de": decoder.state_dict(),
                "en_opt": encoder_optimizer.state_dict(),
                "de_opt": decoder_optimizer.state_dict(),
                "loss": loss,
                "voc_dict": voc.__dict__,
                "embedding": embedding.state_dict()
                }, os.path.join(directory, "{}_{}.tar".format(iteration, "checkpoint")))


class greedy_search_decode(nn.Module):
    def __init__(self, encoder, decoder):
        super(greedy_search_decode, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):
       # forward input through encoder model
       encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
       # prepare encoder's final hidden layer to be first hidden input to the decoder
       decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * SOS_token
       # initialize tensors to append deoded words to
       all_tokens = torch.zeros([0], device=device, dtype=torch.long)
       all_scores = torch.zeros([0], device=device)


