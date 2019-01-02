import argparse
import numpy as np
import pandas as pd
import math
import datetime as dt
import os

home = os.path.expanduser("~")

def createfolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

parser = argparse.ArgumentParser()

parser.add_argument("--study-area", default="Taipei", metavar='', type=str)
parser.add_argument("--radar-folder", default=home+'/OneDrive/01_IIS/04_TY_research/01_Radar_data', metavar='', type=str,
                    help="The folder path of the radar data.")
parser.add_argument("--ty-list", default=home+'/OneDrive/01_IIS/04_TY_research/01_Radar_data/ty_list.xlsx', metavar='', type=str,
                    help="The path of the typhoon list file.")
parser.add_argument("--sta-list", default=home+'/OneDrive/01_IIS/04_TY_research/01_Radar_data/sta_list_all.xlsx', metavar='', type=str,
                    help="The path of the station list file.")
parser.add_argument("--TW-map-file", default=home+'/OneDrive/01_IIS/04_TY_research/01_Radar_data/07_TW_shapefile/gadm36_TWN_2', metavar='',
                    type=str, help="The path of the TW-map file.")

parser.add_argument("--fortran-code-folder", default="fortran_codes/", metavar='', type=str, help="The path of the fortran-code folder")

parser.add_argument("--origin-files-folder", default="/ubuntu_hdd/research/origianal_radar_data_2012-2018", metavar='', type=str,
                    help="The path of the original files folder")
parser.add_argument("--compressed-files-folder", default=home+"/OneDrive/01_IIS/04_TY_research/01_Radar_data/01_compressed_files",
                    metavar='', type=str, help="The path of the compressed files folder")
parser.add_argument("--numpy-files-folder", default=home+"/OneDrive/01_IIS/04_TY_research/01_Radar_data/02_numpy_files", metavar='', type=str,
                    help="The path of the numpy files folder")
parser.add_argument("--figures-folder", default=home+"/OneDrive/01_IIS/04_TY_research/01_Radar_data/03_figures", metavar='', type=str,
                    help="The path of the numpy files folder")


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
parser.add_argument("--F-lon-h", default=121.8875, type=float, metavar='',
                    help='The highest longitude of the forecast frames')

parser.add_argument("--origin-lat-l", default=20, type=float)
parser.add_argument("--origin-lat-h", default=27, type=float)
parser.add_argument("--origin-lon-l", default=118, type=float)
parser.add_argument("--origin-lon-h", default=123.5, type=float)

parser.add_argument("--resolution", default=0.0125, type=float)

args = parser.parse_args()

args.input_size=(math.ceil((args.I_lon_h-args.I_lon_l)/args.resolution)+1,math.ceil((args.I_lat_h-args.I_lat_l)/args.resolution)+1)
args.forecast_size=(math.ceil((args.F_lon_h-args.F_lon_l)/args.resolution)+1,math.ceil((args.F_lat_h-args.F_lat_l)/args.resolution)+1)
args.origin_lat_size=math.ceil((args.origin_lat_h-args.origin_lat_l)/args.resolution)+1
args.origin_lon_size=math.ceil((args.origin_lon_h-args.origin_lon_l)/args.resolution)+1


if __name__ == "__main__":
    print(args.origin_lat_size)
