import argparse

parser = argparse.ArgumentParser(description="parse parameter")
parser.add_argument('-e','--epoch',default=1, type=int, help='epochs')
parser.add_argument('-b','--batch',default=1, type=int, help='batch size')
parser.add_argument('-lr','--lrate',default=1e-3, type = float, help='learning rate')
parser.add_argument('-fi','--file_size',default='small', type=str,help='file size')
parser.add_argument('-dn','--download', default='True', type=str, help='download files')
args = parser.parse_args()