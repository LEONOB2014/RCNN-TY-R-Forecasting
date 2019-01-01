import argparse
import math
import torch
import os

def createfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

parser = argparse.ArgumentParser("This is the argument controller.")

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--gpu', default=0, type=int, metavar='',
                    help='GPU device (default=0)')

parser.add_argument('--max-epochs', default=50, type=int, metavar='',
                    help='Max epochs (default=50)')
parser.add_argument('--batch-size', default=4, type=int, metavar='',
                    help='Batch size (default=4)')
parser.add_argument('--lr', default=1e-4, type=float, metavar='',
                    help='Learning rate (default=1e-4)')
parser.add_argument('--weight-decay', default=1e-2, type=float, metavar='',
                    help='Weight decay (default=1e-2)')
parser.add_argument('--clip-max-norm', default=5, type=float, metavar='',
                    help='Clip max norm (default=5)')

parser.add_argument("--root-dir", metavar='', help='The folder path of the Radar numpy data')
parser.add_argument("--ty-list-file", metavar='', help='The path of ty_list.xlsx')
parser.add_argument("--result-dir", metavar='', help='The path of result folder')

parser.add_argument("--lat-l", default=24.6625, type=float, metavar='',
                    help='Set the lowest latitude value of the study area')
parser.add_argument("--lon-l", default=121.15, type=float, metavar='',
                    help='Set the lowest longitude value of the study area')
parser.add_argument("--lat-h", default=25.4, type=float, metavar='',
                    help='Set the highest latitude value of the study area')
parser.add_argument("--lon-h", default=121.8875, type=float, metavar='',
                    help='Set the highest longitude value of the study area')

parser.add_argument("--res-degree", default=0.0125, metavar='', type=float,
                    help='The resolution degree of the data')

args = parser.parse_args()

args.device = None
args.file_shape = None

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:{:02d}'.format(args.gpu))
else:
    args.device = torch.device('cpu')

args.data_shape = (math.ceil((args.lat_h-args.lat_l)/args.res_degree),
                    math.ceil((args.lon_h-args.lon_l)))

if __name__ == '__main__':
    print(args.root_dir)
