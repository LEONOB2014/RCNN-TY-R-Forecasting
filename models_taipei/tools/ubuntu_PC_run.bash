# I size = 60, F size = 60
python datasetGRU.py \
    --root-dir /home/jack/OneDrive/01_IIS/04_TY_research/01_Radar_data/02_numpy_files \
    --ty-list-file /home/jack/OneDrive/01_IIS/04_TY_research/ty_list.xlsx \
    --result-dir /home/jack/OneDrive/01_IIS/04_TY_research/05_results \
    --I-lat-l 24.55 --I-lat-h 25.4375 --I-lon-l 121.15 --I-lon-h 122.0375 \
    --F-lat-l 24.55 --F-lat-h 25.4375 --F-lon-l 121.15 --F-lon-h 122.0375 \
    --weight-decay 0.1 --gpu 0
