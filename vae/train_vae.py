import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchtext import data, datasets
from model import EncoderRNN, DecoderRNN, VAE


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=20,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=8,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=1.0,
                   help='initial learning rate')
    return p.parse_args()

def train_vae(args):
    hidden_size = 300
    embed_size = 50
    kld_start_inc = 2
    kld_weight = 0.05
    kld_max = 0.1
    kld_inc = 0.000002
    temperature = 0.9
    temperature_min = 0.5
    temperature_dec = 0.000002
    USE_CUDA = torch.cuda.is_available()
    print_loss_total = 0

    print("[!] preparing dataset...")
    TEXT = data.Field(lower=True, fix_length=30)
    LABEL = data.Field(sequential=False)
    train, test = datasets.IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train, max_size=250000)
    LABEL.build_vocab(train)
    train_iter, test_iter = data.BucketIterator.splits(
            (train, test), batch_size=args.batch_size, repeat=False)
    vocab_size = len(TEXT.vocab) + 2

    print("[!] Instantiating models...")
    encoder = EncoderRNN(vocab_size, hidden_size, embed_size, n_layers=1, use_cuda=USE_CUDA)
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, n_layers=2, use_cuda=USE_CUDA)
    vae = VAE(encoder, decoder)
    optimizer = optim.Adam(vae.parameters(), lr=args.lr)
    vae.train()
    if USE_CUDA:
        print("[!] Using CUDA...")
        vae.cuda()

    for epoch in range(1, args.epochs+1):
        for b, batch in enumerate(train_iter):
            x, y = batch.text, batch.label
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            optimizer.zero_grad()

            m, l, z, decoded = vae(x, temperature)
            if temperature > temperature_min:
                temperature -= temperature_dec
            recon_loss = F.cross_entropy(decoded.view(-1, vocab_size), x.contiguous().view(-1))
            kl_loss = -0.5 * (2 * l - torch.pow(m, 2) - torch.pow(torch.exp(l), 2) + 1)
            kl_loss = torch.clamp(kl_loss.mean(), min=0.2).squeeze()
            loss = recon_loss + kl_loss * kld_weight

            if epoch > 1 and kld_weight < kld_max:
                kld_weight += kld_inc

            loss.backward()
            ec = nn.utils.clip_grad_norm(vae.parameters(), args.grad_clip)
            optimizer.step()

            sys.stdout.write('\r[%d] [loss] %.4f - recon_loss: %.4f - kl_loss: %.4f - kld-weight: %.4f - temp: %4f'
                             % (b, loss.data[0], recon_loss.data[0], kl_loss.data[0], kld_weight, temperature))
            print_loss_total += loss.data[0]
            if b % 200 == 0 and b != 0:
                print_loss_avg = print_loss_total / 200
                print_loss_total = 0
                print("\n[avg loss] - ", print_loss_avg)
                _, sample = decoded.data.cpu()[:,0,:].topk(1)
                print("[ORI]: ", " ".join([TEXT.vocab.itos[i] for i in x.data[:,0]]))
                print("[GEN]: ", " ".join([TEXT.vocab.itos[i] for i in sample.squeeze()]))
        torch.save(vae, './snapshot/vae_{}.pt'.format(epoch))


if __name__ == "__main__":
    try:
        args = parse_arguments()
        train_vae(args)
    except KeyboardInterrupt as e:
        print("[STOP]", e)
