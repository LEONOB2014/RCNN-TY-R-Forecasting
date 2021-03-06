import argparse
import torch
import math
import os

def createfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

parser = argparse.ArgumentParser("")

parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--gpu', default=0, type=int, metavar='',
                    help='Set the gpu device(default=0)')


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
parser.add_argument("--ty-list-file", metavar='', help='The path of ty_list excel file')
parser.add_argument("--result-dir", metavar='', help='The path of result folder')


parser.add_argument("--input-lat-l", default=23.5125, type=float, metavar='',
                    help='The lowest latitude of the input frames')
parser.add_argument("--input-lat-h", default=26, type=float, metavar='',
                    help='The highest latitude of the input frames')
parser.add_argument("--input-lon-l", default=120.2125, type=float, metavar='',
                    help='The lowest longitude of the input frames')
parser.add_argument("--input-lon-h", default=122.7, type=float, metavar='',
                    help='The highest longitude of the input frames')

parser.add_argument("--forecast-lat-l", default=24.55, type=float, metavar='',
                    help='The lowest latitude of the forecast frames')
parser.add_argument("--forecast-lat-h", default=25.4375, type=float, metavar='',
                    help='The highest latitude of the forecast frames')
parser.add_argument("--forecast-lon-l", default=121.15, type=float, metavar='',
                    help='The lowest longitude of the forecast frames')
parser.add_argument("--forecast-lon-h", default=122.0375, type=float, metavar='',
                    help='The highest longitude of the forecast frames')

parser.add_argument("--res-degree", default=0.0125, metavar='', type=float,
                    help='The resolution degree of the data')


args = parser.parse_args()

args.device = None
args.file_shape = None

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:{:02d}'.format(args.gpu))
else:
    args.device = torch.device('cpu')

args.forecast_shape = (math.ceil((args.forecast_lat_h-args.forecast_lat_l)/args.res_degree), 
                       math.ceil((args.forecast_lon_h-args.forecast_lon_l)))
args.input_shape = (math.ceil((args.input_lat_h-args.input_lat_l)/args.res_degree), 
                    math.ceil((args.input_lon_h-args.input_lon_l)))
