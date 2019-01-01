# input size = 180, forecast size = 60
python ../../convGRUv2/convGRU_run.py \
    --root-dir /storage_sdd/kun/OneDrive/01_IIS/04_TY_research/01_Radar_data/02_wrangled_files_Taipei_I.60_F.60 \
    --ty-list-file /storage_sdd/kun/OneDrive/01_IIS/04_TY_research/ty_list.xlsx \
    --result-dir /storage_sdd/kun/OneDrive/01_IIS/04_TY_research/05_results \
    --weight-decay 0.1  --gpu 1
