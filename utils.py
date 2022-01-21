import os.path

from torch.utils.data import Dataset

class DownLoad_ataset:

    def __init__(self,
                 root_dir:str='datatset',
                 file_size:str=small,
                 download:bool=True,):

        if file_size == 'small':
            self.url  = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/data'
            self.dir = os.path.join(root_dir,file_size)
            self.file_list = ['dict.en-vi','train.en','train.vi','tst2012.en','tst2012.vi','tst2013.en','tst2013.vi','vocab.en','vocab.vi']
            print('Dataset\'s is about WMT\'15 English-Czech data data| data[small]')

        elif file_size =='medium':
            self.url = 'https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/data'
            self.dir = os.path.join(root_dir,file_size)
            print('Dataset\'s is about WMT\'14 English-German data| data[medium]')

        elif file_size == 'large':
            self.url = 'https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-v/data'
            self.dir = os.path.join(root_dir,file_size)
            print('Dataset\'s is about WMT\'15 English-Czech | data[large]')

        if download:
            self._Download()

    # def _Download(self):


