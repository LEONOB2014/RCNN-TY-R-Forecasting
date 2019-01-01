# input size = 60, forecast size = 60
python ../../convGRUv2/convGRU_run.py \
    --root-dir /home/jack/storage_ssd/OneDrive/01_IIS/04_TY_research/01_Radar_data/02_wrangled_files_Taipei_I.60_F.60 \
    --ty-list-file /home/jack/storage_ssd/OneDrive/01_IIS/04_TY_research/ty_list.xlsx \
    --result-dir /home/jack/storage_ssd/OneDrive/01_IIS/04_TY_research/05_results \
    --input-lat-l 24.55 --input-lat-h 25.4375 --input-lon-l 121.15 --input-lon-h 122.0375 \
    --forecast-lat-l 24.55 --forecast-lat-h 25.4375 --forecast-lon-l 121.15 --forecast-lon-h 122.0375 \
    --weight-decay 0.1 --gpu 0
