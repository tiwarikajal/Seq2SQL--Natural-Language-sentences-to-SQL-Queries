from __future__ import unicode_literals, print_function, division
from util.DataConversionUtil import DataConversionUtil
from util.LanguageUtil import prepareData, tensorsFromPair, tensorFromSentence, prepareValData
import random
import torch
import torch.nn as nn
from torch import optim
from util.graph_plotter import showPlot
from baseline.model.encoder import EncoderRNN
from baseline.model.decoder import DecoderRNN
from baseline.model.decoder import AttnDecoderRNN
from util.constants import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer,
          criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)

        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True  # if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def hyperparam(hidden_size):

    global input_lang
    global output_lang
    global pairs
    input_lang, output_lang, pairs = prepareValData("en", "sql")
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    lr = [0.0001, 0.001, 0.01, 0.1, 1]
    high = 0
    best_lr = 0
    for l in lr:
        print("Searching for best params...")
        trainIters(encoder1, attn_decoder1, 250000, print_every=1000, plot_every=1000,learning_rate=l) #Change number of iter
        accuracy = evaluateRandomly(encoder1,attn_decoder1, n=10)
        if(accuracy > high):
            best_lr = l
    return best_lr


def trainIters(encoder, decoder, n_iters, print_every=10, plot_every=20, learning_rate=0.0005):
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs), input_lang, output_lang) for i in range(n_iters)]
    criterion = nn.CrossEntropyLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%d %d%%) %.4f' % (iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses, "Baseline loss")


def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=1000):
    correct = 0
    for i in range(n):
        pair = random.choice(pairs)
        print('\nEnglish Question-', pair[0])
        print('Ground truth Query-', pair[1])
        generated_tokens, attentions = evaluate(encoder, decoder, pair[0])
        generated_query = ' '.join(generated_tokens)
        if generated_query[:-6] == pair[1]:
            correct += 1
        print('Generated Query-', generated_query)
    print("\n\nCorrect Examples : {} out of {}".format(correct, n))
    return correct / n * 100


def run_baseline():
    hidden_size = 256
    x = DataConversionUtil()
    lr_best = hyperparam(hidden_size)
    
    global input_lang
    global output_lang
    global pairs
    input_lang, output_lang, pairs = prepareData("en", "sql")
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder1 = DecoderRNN(hidden_size, output_lang.n_words).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)
    trainIters(encoder1, attn_decoder1, 250000, print_every=1000, plot_every=1000,learning_rate=lr_best)
    acc = evaluateRandomly(encoder1, attn_decoder1, n=1000)
    print("Accuracy achieved:", acc)


if __name__ == '__main__':
    run_baseline()