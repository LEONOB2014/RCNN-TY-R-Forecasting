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
parser.add_argument("--study-area", default="Taipei", type=str)
parser.add_argument("--ty-list", default='../ty_list.xlsx', type=str)
parser.add_argument("--sta-list", default='../sta_list_all.xlsx', type=str)
parser.add_argument("--TW-map-file", default="../TW_shapefile/gadm36_TWN_2", type=str)

parser.add_argument("--original-files-folder", default="/ubuntu_hdd/01_research/origianal_radar_data_2012-2018", type=str)
parser.add_argument("--compressed-files-folder", default=home+"/Desktop/01_compressed_files", type=str)
parser.add_argument("--readable-files-folder", default="../01_readable_files", type=str)
parser.add_argument("--wrangled-files-folder", default="../02_wrangled_files", type=str)
parser.add_argument("--wrangled-figs-folder", default="../02_wrangled_fig", type=str)

parser.add_argument("--I-lat-l", default=23.9125, type=float)
parser.add_argument("--I-lat-h", default=26.15, type=float)
parser.add_argument("--I-lon-l", default=120.4, type=float)
parser.add_argument("--I-lon-h", default=122.6375, type=float)

parser.add_argument("--F-lat-l", default=24.6625, type=float)
parser.add_argument("--F-lat-h", default=25.4, type=float)
parser.add_argument("--F-lon-l", default=121.15, type=float)
parser.add_argument("--F-lon-h", default=121.8875, type=float)

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
