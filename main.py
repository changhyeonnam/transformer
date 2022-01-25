import os.path
import torch
import pandas as pd
import numpy as np
import requests
from parser import args
from utils import downloads

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


