import torch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import os

from parser import args
from utils import downloads
from model.Encoder import Encoder
from model.Decoder import Decoder
from model.Seq2Seq import Seq2Seq
import torch.nn as nn
import random
import math
import time

# argparse doesn't support boolean  type
download_files = True if args.download == 'True' else False

# gpu info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
if device == 'cuda':
    print("Current cuda device : ",torch.cuda.current_device())
    print("Count of Using GPUS : ",torch.cuda.device_count())

# file_list
if args.file_size == 'small':
    file_list = ['dict.en-vi','train.en','train.vi','tst2012.en','tst2012.vi','tst2013.en','tst2013.vi','vocab.en','vocab.vi']
elif args.file_size == 'medium':
    file_list = ['dict.en-de','newstest2012.de','newstest2012.en','newstest2013.de','newstest2013.en','newstest2014.de','newstest2014.en','newstest2015.de','newstest2015.en','train.align','train.de','train.en','vocab.50K.de','vocab.50K.en']
elif args.file_size == 'large':
    file_list = ['dict.en-cs','newstest2013.cs','newstest2013.en','newstest2014.cs','newstest2014.en','newstest2015.cs','newstest2015.en','newstest2015.de','train.cs','train.en','vocab.50K.cs','vocab.50K.en','vocab.1K.cs','vocab.1K.en','vocab.10K.cs','vocab.10K.en','vocab.20K.cs','vocab.20K.en','vocab.100K.cs','vocab.100K.en','vocab.char.200.cs','vocab.char.200.en']

# root directory for dataset
root ='dataset'

# Download dataset
if download_files:
    downloads(file_list=file_list,root_dir=root,file_size=args.file_size)
else:
    print("Dataset already downloaded")

eng = os.path.join(root,args.file_size)
vi = os.path.join(root,args.file_size)
with open(os.path.join(eng,'train.en'), 'r') as file :
    src = file.read()
with open(os.path.join(eng,'train.vi'), 'r') as file :
    trg = file.read()

spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')

def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

SRC = Field(tokenize= tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower = True,
            batch_first=True)
TRG = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True,
            batch_first=True)

train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
     batch_size = BATCH_SIZE,
     device = device)
INPUT_DIM = len(SRC.vocab)
OUTPUT_DIM = len(TRG.vocab)
HID_DIM = 256
ENC_LAYERS = 3
DEC_LAYERS = 3
ENC_HEADS = 8
DEC_HEADS = 8
ENC_PF_DIM = 512
DEC_PF_DIM = 512
ENC_DROPOUT = 0.1
DEC_DROPOUT = 0.1

enc = Encoder(input_dim = INPUT_DIM,
              hid_dim = HID_DIM,
              n_layers = ENC_LAYERS,
              n_heads = ENC_HEADS,
              pf_dim = ENC_PF_DIM,
              dropout=ENC_DROPOUT).to(device)

dec = Decoder(OUTPUT_DIM,
              HID_DIM,
              DEC_LAYERS,
              DEC_HEADS,
              DEC_PF_DIM,
              DEC_DROPOUT)

SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX).to(device)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'The model has {count_parameters(model):,} trainable parameters')

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)
model.apply(initialize_weights);
LEARNING_RATE = 0.0005



optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1])

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:, :-1])

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'tut6-model.pt')

    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
