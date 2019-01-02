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


parser.add_argument("--I-lat-l", default=23.9125, type=float, metavar='',
                    help='The lowest latitude of the input frames')
parser.add_argument("--I-lat-h", default=26.15, type=float, metavar='',
                    help='The highest latitude of the input frames')
parser.add_argument("--I-lon-l", default=120.4, type=float, metavar='',
                    help='The lowest longitude of the input frames')
parser.add_argument("--I-lon-h", default=122.6375, type=float, metavar='',
                    help='The highest longitude of the input frames')

parser.add_argument("--F-lat-l", default=24.6625, type=float, metavar='',
                    help='The lowest latitude of the forecast frames')
parser.add_argument("--F-lat-h", default=25.4, type=float, metavar='',
                    help='The highest latitude of the forecast frames')
parser.add_argument("--F-lon-l", default=121.15, type=float, metavar='',
                    help='The lowest longitude of the forecast frames')
parser.add_argument("--F-lon-h", default=121.8875, type=float,
                    help='The highest longitude of the forecast frames')

parser.add_argument("--res-degree", default=0.0125, type=float, metavar='',
                    help='The res_degree degree of the data')


args = parser.parse_args()

args.origin_lat_l = 20
args.origin_lat_h = 27
args.origin_lon_l = 118
args.origin_lon_h = 123.5

args.I_x_left = int((args.I_lon_l-args.origin_lon_l)/args.res_degree + 1)
args.I_x_right = int(args.I_x_left + (args.I_lon_h-args.I_lon_l)/args.res_degree + 1)
args.I_y_low = int((args.I_lat_l-args.origin_lat_l)/args.res_degree + 1)
args.I_y_high = int(args.I_y_low + (args.I_lat_h-args.I_lat_l)/args.res_degree + 1)

args.F_x_left = int((args.F_lon_l-args.origin_lon_l)/args.res_degree + 1)
args.F_x_right = int(args.F_x_left + (args.F_lon_h-args.F_lon_l)/args.res_degree + 1)
args.F_y_low = int((args.F_lat_l-args.origin_lat_l)/args.res_degree + 1)
args.F_y_high = int(args.F_y_low + (args.F_lat_h-args.F_lat_l)/args.res_degree + 1)

args.device = None
args.file_shape = None

if not args.disable_cuda and torch.cuda.is_available():
    args.device = torch.device('cuda:{:02d}'.format(args.gpu))
else:
    args.device = torch.device('cpu')

args.forecast_shape = (math.ceil((args.F_lat_h-args.F_lat_l)/args.res_degree),
                       math.ceil((args.F_lon_h-args.F_lon_l)/args.res_degree))
args.input_shape = (math.ceil((args.I_lon_h-args.I_lon_l)/args.res_degree),
                    math.ceil((args.I_lon_h-args.I_lon_l)/args.res_degree))

if __name__ == "__main__":
    print(args.I_x_left)
    print(args.I_x_right)
    print(args.I_y_low)
    print(args.I_y_high)
