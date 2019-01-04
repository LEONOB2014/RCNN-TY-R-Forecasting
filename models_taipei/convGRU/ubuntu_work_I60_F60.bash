# I size = 60, F size = 60
python convGRU_run.py \
    --root-dir /home/jack/OneDrive/01_IIS/04_TY_research/01_Radar_data/02_numpy_files \
    --ty-list-file /home/jack/OneDrive/01_IIS/04_TY_research/ty_list.xlsx \
    --result-dir /home/jack/OneDrive/01_IIS/04_TY_research/05_results \
    --I-lat-l 24.6625 --I-lat-h 25.4 --I-lon-l 121.15 --I-lon-h 121.8875 \
    --F-lat-l 24.6625 --F-lat-h 25.4 --F-lon-l 121.15 --F-lon-h 121.8875 \
    --weight-decay 0.1 --gpu 0
