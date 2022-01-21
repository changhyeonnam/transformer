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
root ='dataset'


# Download dataset
if download_files:
    downloads(root_dir=root,file_size=args.file_size)