import os.path

import torch
import pandas as pd
import numpy as np
import requests

# gpu info
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'device:{device}')
if device == 'cuda':
    print("Current cuda device : ",torch.cuda.current_device())
    print("Count of Using GPUS : ",torch.cuda.device_count())

url = "https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi"
dir = "dataset"
if not os.path.exists(dir):
    os.makedirs(dir)

for file in li:
    file_url = os.path.join(url,file)
    req = requests.get(file_url, stream=True)
    with open(os.path.join(dir,file), mode="wb") as fd:
        for chunck in req.iter_content(chunk_size=None):
            fd.write(chunck)
