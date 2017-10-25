import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

SOS = 0

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 n_layers=1, dropout_p=0.1, use_cuda=True):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda
        self.embed = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers,
                          dropout=dropout_p, bidirectional=True)
        self.o2p = nn.Linear(hidden_size, output_size * 2)

    def sample(self, mu, logvar):
        eps = Variable(torch.randn(mu.size()))
        if self.use_cuda:
            eps = eps.cuda()
        std = torch.exp(logvar / 2.0)
        return mu + eps * std

    def forward(self, x):
        embedded = self.embed(x)
        output, hidden = self.gru(embedded, None)
        output = output[-1]
        # Sum bidirectional outputs
        output = output[:, :self.hidden_size] + output[: ,self.hidden_size:]
        ps = self.o2p(output)
        mu, logvar = torch.chunk(ps, 2, dim=1)
        z = self.sample(mu, logvar)
        return mu, logvar, z

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers*2, batch_size, self.hidden_size))
        if self.use_cuda:
            hidden = hidden.cuda()
        return hidden

class DecoderRNN(nn.Module):
    """
    Decode from z into sequence using RNN
    """
    def __init__(self, input_size, hidden_size, output_size,
                 n_layers=1, dropout_p=0.1, use_cuda=True):
        super(DecoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.use_cuda = use_cuda

        self.embed = nn.Embedding(output_size, hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.z2h = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size + input_size, hidden_size, n_layers,
                          dropout=dropout_p)
        self.i2h = nn.Linear(hidden_size + input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size + input_size, output_size)

    def _sample(self, output, temp):
        _, topi = output.data.topk(1)
        x = Variable(topi.squeeze())
        return x

    def _step(self, z, x, h):
        x = F.relu(self.embed(x))
        x = torch.cat((x, z), 1)
        x = x.unsqueeze(0)
        o, h = self.gru(x, h)
        o = o.squeeze(0)
        o = torch.cat((o, z), 1)
        o = self.out(o)
        return o, h

    def forward(self, z, xs, temp=None, timesteps=None):
        timesteps = timesteps if timesteps else xs.size(0)
        batch_size = xs.size(1)
        outputs = Variable(torch.zeros(timesteps, batch_size, self.output_size))
        x = Variable(torch.LongTensor([SOS] * batch_size))
        if self.use_cuda:
            outputs = outputs.cuda()
            x = x.cuda()
        hidden = self.z2h(z).unsqueeze(0).repeat(self.n_layers, 1, 1)
        for i in range(timesteps):
            output, hidden = self._step(z, x, hidden)
            outputs[i] = output
            teacher_forcing = random.random() < temp if temp else None
            x = xs[i] if teacher_forcing else self._sample(output, temp)
        return outputs.squeeze(1)


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, xs, temp=1.0):
        m, l, z = self.encoder(xs)
        decoded = self.decoder(z, xs, temp)
        return m, l, z, decoded
