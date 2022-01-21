import os
from torch.utils.data import Dataset
import requests

def downloads(file_list,root_dir:str='datatset',file_size:str='small',):

    global url, dir
    if file_size == 'small':
        url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi'
        dir = os.path.join(root_dir, file_size)
        print('Downloading Dataset\'s is about WMT\'15 English-Czech data | data size\'s[small]')

    elif file_size == 'medium':
        url = 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de'
        dir = os.path.join(root_dir, file_size)
        print('Downloading Dataset\'s is about WMT\'14 English-German data| data size\'s[medium]')

    elif file_size == 'large':
        url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-v'
        dir = os.path.join(root_dir, file_size)
        print('Downloading Dataset\'s is about WMT\'15 English-Czech | data size\'s[large]')

    if not os.path.exists(dir):
        os.makedirs(dir)

    for file in file_list:
        file_url = os.path.join(url,file)
        req = requests.get(file_url,stream=True)
        with open(os.path.join(dir,file),mode='wb') as fd:
                for chunck in req.iter_content(chunk_size=None):
                    fd.write(chunck)

    print("Downloading Dataset complete")



