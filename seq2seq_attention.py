"""
we train the model sentence by sentence, i.e., setting the batch_size = 1
"""
from __future__ import unicode_literals, print_function, division

import argparse
import logging
import math
import random
import time
from io import open

import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15 #512??????????????????????????????
teacher_forcing_ratio = 0.5


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """

    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"), 
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab


######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################


class EncoderRNN(nn.Module):
    """the class for the enoder RNN"""

    def __init__(self, input_size, hidden_size):
        # input_size: src_side vocabulary size
        # hidden_size: hidden state dimension
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # TODO 1: Initilize your word embedding, encoder rnn
        self.embedding = nn.Embedding(input_size, hidden_size) #embedded input
        self.gru = nn.GRU(hidden_size, hidden_size) #creating the neural network

    def forward(self, input, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        # TODO 2: complete the forward computation, given the input and the previous hidden state
        embedded = self.embedding(input)
        # reshaping the tensor
        embedded = torch.reshape(embedded, [1, 1, -1]) #reshape the embeddings for calculation
        output = embedded
        output, hidden = self.gru(output, hidden) #forward the input to the neural network
        # print('encoder_out: ' + str(output.size()))
        # print('encoder_hid: ' + str(hidden.size()))
        return output, hidden
        #  return the output and the hidden state


    def get_initial_hidden_state(self):
        # NOTE: you need to change here if you use LSTM as the rnn unit
        return torch.zeros(1, 1, self.hidden_size, device=device)


class AttentionDot(nn.Module):
    """the class for general attention with dot product"""
    def __init__(self, hidden_size):
        super(AttentionDot, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, output_enc, hidden_dec):
        # shapes: output_enc (1, len_src, hidden_size); hidden_dec ((1, 1, hidden_size))
        # 1. compute the attention weights; 2. compute the context vector
        # hidden_size = 512
        # print('AttnDot_hidden_dec: ' + str((hidden_dec).size())) # ([1,1,512]) -- if with squeeze(0): [1,512]
        # print('AttnDot_output_dec: ' + str((output_enc.transpose(1,2)).size())) # ([1,2,512]) -- if with squeeze(0): [2,512]
        # print('AttnDot_output_dec' + str(output_enc))
        # both batches (hidden_dec and output_dec) for bmm should be 3D tensors
        scores = torch.bmm(hidden_dec, output_enc.transpose(1,2)) # shape of scores: (1,1,len_src)
        attn_weights = self.softmax(scores)  # shape: (1,1,len_src) #apply softmax to get the weight
        ctx_vec = torch.bmm(attn_weights, output_enc) #apply the weights to the output and get the context vector
        return ctx_vec


class AttentionGeneral(nn.Module): #Luong
    """the class for general attention with general computation"""
    def __init__(self, hidden_size):
        super(AttentionGeneral, self).__init__()
        # TODO 7 complete the attention with general computation
        self.attention = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, output_enc, hidden_dec):
        # shapes: output_enc (1, len_src, hidden_size); hidden_dec ((1, 1, hidden_size))
        # 1. compute the attention weights; 2. compute the context vector
        # TODO 7 complete the attention with general computation
        # print('attn_hidden: ' + str(self.attention(hidden_dec).size()))
        # print('output_enc: ' + str((output_enc).size())) #output_enc.size() = ([1, 15, 512])
        scores = torch.bmm((self.attention(hidden_dec)), output_enc.transpose(1,2)) #first pass the hidden through a linear layer, and matrix computation with output
        attn_weights = F.softmax(scores.view(1,-1), dim = 1) #apply softmax to get the weights
        ctx_vec = torch.bmm(attn_weights.unsqueeze(0), output_enc) #apply the weights to the output and get the context vector
        return ctx_vec


class AttentionConcat(nn.Module): #Bahdanau
    """the class for general attention with concat computation"""
    def __init__(self, hidden_size):
        super(AttentionConcat, self).__init__()
        # TODO 8 complete the attention with concat computation
        self.attention = nn.Linear(hidden_size, hidden_size)
        self.vt = nn.Parameter(torch.FloatTensor(1, hidden_size)) #intermediate parameter vector
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, output_enc, hidden_dec):
        # shapes: output_enc (1, len_src, hidden_size); hidden_dec ((1, 1, hidden_size))
        # 1. compute the attention weights; 2. compute the context vector
        # TODO 8 complete the attention with concat computation
        scores = F.tanh(self.attention(hidden_dec + output_enc)) #concatenate hidden and output and pass through the linear layer
        # print('concat_scores_shape: ' + str(scores.size()))
        attn_weights = self.softmax(torch.bmm(scores, self.vt.unsqueeze(-1)).view(1,-1)) #apply softmac to the dot product of the parameter v and scores to get weights
        # attn_weights = self.softmax(scores)
        # print('w_view(1,-1)_shape: ' + str((self.softmax(torch.bmm(scores, self.vt.unsqueeze(-1)).view(1,-1))).size()))
        # print('concat_attn_weights: ' + str(attn_weights.size()))
        ctx_vec = torch.bmm(attn_weights.unsqueeze(0), output_enc) #apply the weights to the output and get the context vector
        return ctx_vec

class AttentionMultihead(nn.Module): #optional
    """the class for multi-head attention"""
    def __init__(self, hidden_size, num_head):
        super(AttentionMultihead, self).__init__()
        self.hidden_size = hidden_size
        self.num_head = num_head
        self.dim_head = hidden_size // num_head
        # TODO 9 complte the multihead attention with scaled-dot product

    def forward(self, output_enc, hidden_dec):
        # shapes: output_enc (1, len_src, hidden_size); hidden_dec ((1, 1, hidden_size))
        # 1. compute the context vector for each head; 2. concat context vectors from all heads
        # TODO 9 complte the multihead attention with scaled-dot product
        return ctx_vec


class AttnDecoderRNN(nn.Module):
    """the class for the decoder with attention"""

    def __init__(self, hidden_size, output_size, attn_type, num_head):
        # hidden_size: hidden state dimension
        # output_size: trg_side vocabulary size
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        # here we set attention types based on the parameter "attn_type"
        if attn_type == "dot":
            self.attn = AttentionDot(hidden_size)
        elif attn_type == "general":
            self.attn = AttentionGeneral(hidden_size)
        elif attn_type == "concat":
            self.attn = AttentionConcat(hidden_size)
        elif attn_type == "multihead":
            self.attn = AttentionMultihead(hidden_size, num_head)
        # TODO 3: Initilize your word embedding, decoder rnn, output layer, softmax layer
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        # an extra linear layer to transform the vector which is the combination of the target-side input embedding and the context vector passed from the attention
        # output of the hidden layer and context vector is of the same size, thus times 2, creating a matrix here
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        # NOTE: compared to assignment 4, you need an extra linear layer to transform
        # the vector which is the combination of the target-side input embedding and the context vector passed from the attention
        

    def forward(self, input, output_enc, hidden_dec):
        """runs the forward pass of the decoder
        returns the probability distribution, hidden state
        """
        # TODO 4: complete the forward computation, given the input and the previous hidden state
        embedded = self.embedding(input).view(1,1,-1)
        # print('output_enc: ' + str(output_enc.size())) #output_enc = torch.Size([1, 15, 512])
        # print('embedded_size: ' + str(embedded.size())) #embedded_size = ([1, 1, 512])
        # print('hidden_dec: ' + str(hidden_dec.size())) #hidden_dec.size = ([1, 1, 512])
        # embedded[0] = shape [1,512], hidden_dec[0] = shape [1,512]
        attn_weights = F.softmax(self.attn(output_enc, hidden_dec), dim = 1) #compute the weights
        # print('attn_weights' + str(attn_weights.size())) #attn_weights_Size = ([1, 1, 512])
        # you need to consider the context vector passed from the attn function
        probs = torch.cat((attn_weights[0], embedded[0]), dim = 1).unsqueeze(0) #concatenate the weights and embeded and reshape for next step
        probs = self.attn_combine(probs) #pass the weight vector through the linear layer
        probs = F.relu(probs) #softmax to convert the weight into probabilities
        probs, hidden_dec = self.gru(probs, hidden_dec) 
        probs = F.log_softmax(self.out(probs[0]), dim = 1) # softmax to get the probabilities
        # return the following variables
        # probs: the output after applying LogSoftmax function
        # and hidden: hidden states
        return probs, hidden_dec, attn_weights

    def get_initial_hidden_state(self):
        # NOTE: you need to change here if you use LSTM as the rnn unit
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion):
    encoder_hidden = encoder.get_initial_hidden_state()
    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()
    optimizer.zero_grad()

    input_length = input_tensor.size(0) #8
    target_length = target_tensor.size(0)

    loss = 0
    # encoder-side forward computation
    # Now you need to store all the encoder_output, and pass them to the attention
    encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
    # print('beforetrain_input_length: ' + str(input_length))
    # print('beforetrain_encoder_outputs: ' + str(encoder_outputs.size()))
    for ei in range(input_length):
        # TODO 5: feed each input to the encoder, and get the output
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0,0]

    encoder_outputs = encoder_outputs.unsqueeze(0)
    # print('train_encoder_outputs: ' + str(encoder_outputs.size())) #train_encoder_outputs: torch.Size([15, 512]), with unsqueeze(0):([1, 15, 512])
    #  set the first input to the decoder is the symbol "SOS"
    decoder_input = torch.tensor([[SOS_index]], device=device)
    # TODO 5: initialize the decoder with the last encoder hidden state
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    # print(decoder_input.type())
    # print(encoder_outputs.type())
    # print(decoder_hidden.type())
    # target-side generation
    # print('train_decoder_input: ' + str(decoder_input.size())) #torch.Size([1, 1])
    # print('train_decoder_encoderoutputs: ' + str(encoder_output.size())) #torch.Size([1, 1, 512])
    # print('train_decoder_decoderhidden: ' + str(decoder_hidden.size())) #torch.Size([1, 1, 512])
    for di in range(target_length):
        # TODO 5: get the output of the decoder, for each step
        # Note: here is different from assignment 4
        decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, encoder_outputs, decoder_hidden)
        # TODO 5: compute the loss
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()
        loss += criterion(decoder_output, target_tensor[di])

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            decoder_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            if decoder_input.item() == EOS_token:
                break

    #  back-propagation step, torch helps you do it automatically
    loss.backward()
    #  update parameters, the optimizer will help you automatically
    optimizer.step()

    loss = loss.item() / target_length  # average of all the steps
    return loss

######################################################################


def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """ runs translation, returns the output """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()

        encoder_outputs = torch.zeros(MAX_LENGTH, encoder.hidden_size, device=device)
        for ei in range(input_length):
            # TODO 6: feed each input to the encoder, and get the output
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden) #encoder forward input, hidden
            # encoder_outputs[ei] += encoder_output[0,0]

        encoder_outputs = encoder_outputs.unsqueeze(0)

        #  set the first input to the decoder is the symbol "SOS"
        decoder_input = torch.tensor([[SOS_index]], device=device)
        # TODO 6: initialize the decoder with the last encoder hidden state
        decoder_hidden = encoder_hidden
        decoded_words = []

        for di in range(max_length):
            # TODO 6: get the output of the decoder, for each step
            decoder_output, decoder_hidden, decoder_attn = decoder(decoder_input, encoder_outputs, decoder_hidden)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words

######################################################################


# Translate (dev/test)set takes in a list of sentences and writes out their translates
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=512, type=int, #hidden_size default = 256
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int, #default = 100000
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--status_every', default=500, type=int,
                    help='print how many examples have been learned ')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.0001, type=float, #default = 0.001
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')
    ap.add_argument('--inference', action='store_true')
    ap.add_argument('--attn_type', default='dot',
                    help='attention types: dot, general, concat, or multihead')
    ap.add_argument('--attention_head', default=1, type=int,
                    help='the number of head in multi-head attention')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration 
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)
    # print('src_vocab' + str(src_vocab))
    # TODO 0: initialize the encoder and the decoder here ()
    hidden_size = 512
    embed_size = 256
    num_layers = 1
    attn_type = 'concat' #parameters to be changed for testing

    encoder = EncoderRNN(src_vocab.n_words, hidden_size)
    decoder = AttnDecoderRNN(hidden_size, tgt_vocab.n_words, attn_type, num_head = 1) 

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    if args.load_checkpoint is not None and args.inference:
        translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

        references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
        candidates = [clean(sent).split() for sent in translated_sentences]
        test_bleu = corpus_bleu(references, candidates)
        logging.info('Test BLEU score: %.2f', test_bleu)
        return

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    # start training
    while iter_num < args.n_iters:
        iter_num += 1
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.status_every == 0:
            logging.info('has learnt %d examples', iter_num)
        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

if __name__ == '__main__':
    main()
