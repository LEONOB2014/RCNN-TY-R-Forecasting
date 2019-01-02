from args_tools import *

def data_wrangling():
    print("*" * 20)
    print("*{:^18s}*".format('Data Wrangler'))
    print("*" * 20)
    print("-" * 20)
    # Taipei
    study_area = args.study_area
    flag = "I.{:d}_F.{:d}".format(args.input_size[0],args.forecast_size[0])

    # Set path
    readable_files_folder = args.readable_files_folder
    wrangled_files_folder = args.wrangled_files_folder+"_"+study_area+"_"+flag

    #original image index
    y_index = np.linspace(args.origin_lat_l,args.origin_lat_h,args.origin_lat_size)
    x_column = np.linspace(args.origin_lon_l,args.origin_lon_h,args.origin_lon_size)

    for i in sorted(os.listdir(readable_files_folder)):
        readable_files_path = readable_files_folder+'/'+i
        print("|{:^18s}|".format(i))
        print("-" * 20)
        tmp = 0
        files_list = sorted(os.listdir(readable_files_path))
        counts=0
        for j in np.arange(len(files_list)):
            if j>0:
                time_step = dt.datetime.strptime(files_list[j][4:-4],"%Y%m%d%H%M")-dt.datetime.strptime(files_list[j-1][4:-4],"%Y%m%d%H%M")
            else:
                time_step = 0

            createfolder(wrangled_files_folder+"/"+files_list[j][:3])
            # set lat and lon
            if files_list[j][:3] == 'RAD':
                lat_l = args.I_lat_l
                lat_h = args.I_lat_h
                lon_l = args.I_lon_l
                lon_h = args.I_lon_h
            else:
                lat_l = args.F_lat_l
                lat_h = args.F_lat_h
                lon_l = args.F_lon_l
                lon_h = args.F_lon_h

            if time_step == dt.timedelta(minutes=20):
                file_in1 = readable_files_path+"/"+files_list[j]
                file_in2 = readable_files_path+"/"+files_list[j-1]
                data1 = pd.read_table(file_in1,delim_whitespace=True,header=None)
                data2 = pd.read_table(file_in2,delim_whitespace=True,header=None)
                data1.columns = x_column
                data1.index = y_index
                data2.columns = x_column
                data2.index = y_index
                data_loc = ((data1+data2)/2).loc[lat_l:lat_h,lon_l:lon_h]
                # output missing file
                file_time = dt.datetime.strftime(dt.datetime.strptime(files_list[j][4:-4],"%Y%m%d%H%M")-dt.timedelta(minutes=10),"%Y%m%d%H%M")
                outputname = wrangled_files_folder+"/"+files_list[j][:3]+'/'+i+"_"+file_time
                np.save(outputname,data_loc.values)

                # output file
                data_loc = data1.loc[lat_l:lat_h,lon_l:lon_h]
                outputname = wrangled_files_folder+"/"+files_list[j][:3]+'/'+i+files_list[j][3:-4]
                np.save(outputname,data_loc.values)

            else:
                file_in = os.path.join(readable_files_path,files_list[j])
                data = pd.read_table(file_in,delim_whitespace=True,header=None)

                data.columns = x_column
                data.index = y_index

                # extract data with specific lon and lat range
                data_loc = data.loc[lat_l:lat_h,lon_l:lon_h]

                outputname = wrangled_files_folder+"/"+files_list[j][:3]+'/'+i+files_list[j][3:-4]
                np.save(outputname,data_loc.values)
        print(" Wrangle the data sucessfully!")

def check_data():
    # Taipei
    study_area = args.study_area
    flag = "I.{:d}_F.{:d}".format(args.input_size[0],args.forecast_size[0])
    # Set path
    wrangled_files_folder = args.wrangled_files_folder+"_"+study_area+"_"+flag

    ty_list = pd.read_excel(args.ty_list)

    qpe_list = [x[-16:-4] for x in os.listdir(wrangled_files_folder+"/"+"QPE")]
    qpf_list = [x[-16:-4] for x in os.listdir(wrangled_files_folder+"/"+"QPF")]
    rad_list = [x[-16:-4] for x in os.listdir(wrangled_files_folder+"/"+"RAD")]
    qpe_list_miss = []
    qpf_list_miss = []
    rad_list_miss = []
    for i in np.arange(len(ty_list)):
        for j in np.arange(1000):
            time = ty_list.loc[i,"Time of issuing"] + pd.Timedelta(minutes=10*j)
            if time > ty_list.loc[i,"Time of canceling"]:
                break
            time = time.strftime("%Y%m%d%H%M")
            if time not in qpe_list:
                qpe_list_miss.append(time[:4]+"."+ty_list.loc[i,"En name"]+"_"+time+".npy")
            if time not in qpf_list:
                qpf_list_miss.append(time[:4]+"."+ty_list.loc[i,"En name"]+"_"+time+".npy")
            if time not in rad_list:
                rad_list_miss.append(time[:4]+"."+ty_list.loc[i,"En name"]+"_"+time+".npy")
    missfiles = np.concatenate([np.array(qpe_list_miss),np.array(rad_list_miss),np.array(qpf_list_miss)])
    missfiles_index = []
    for i in range(len(qpe_list_miss)):
        missfiles_index.append("QPE")
    for i in range(len(rad_list_miss)):
        missfiles_index.append("RAD")
    for i in range(len(qpf_list_miss)):
        missfiles_index.append("QPF")

    missfiles = pd.DataFrame(missfiles,index=missfiles_index,columns=["File_name"])
    missfiles.to_excel("../Missing_files.xlsx")

    print(" Check the data completely!")

if __name__ == "__main__":
    data_wrangling()
    check_data()
