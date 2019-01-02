import sys
import gzip
from args_tools import *

def extract_original_data():
    #load typhoon list file
    ty_list = pd.read_excel(args.ty_list)

    origin_files_folder = args.origin_files_folder
    compressed_files_folder = args.compressed_files_folder
    if os.path.exists(compressed_files_folder):
        return 'Already extract oringinal data'
    for i in range(len(ty_list)):
        # get the start and end time of the typhoon
        year = ty_list.loc[i,"Time of issuing"].year
        date_s = dt.datetime.strftime(ty_list["Time of issuing"][i] - dt.timedelta(hours=8),format="%Y%m%d")
        date_e = dt.datetime.strftime(ty_list["Time of canceling"][i] - dt.timedelta(hours=8),format="%Y%m%d")
        time_s = dt.datetime.strftime(ty_list["Time of issuing"][i] - dt.timedelta(hours=8),format="%H%M")
        time_e = dt.datetime.strftime(ty_list["Time of canceling"][i] - dt.timedelta(hours=8),format="%H%M")
        ty_name = ty_list.loc[i,"En name"]

        print("|{:8s}| start time: {:s} | end time: {:s} |".format(ty_name+"",date_s,date_e))
        print("|{:8s}|             {:8s} |           {:8s} |".format(" ",time_s,time_e))

        tmp_path1 = os.path.join(origin_files_folder,str(year))
        for j in os.listdir(tmp_path1):
            if dt.datetime.strptime(date_s,"%Y%m%d") <= dt.datetime.strptime(j,"%Y%m%d") <= dt.datetime.strptime(date_e,"%Y%m%d"):
                tmp_path2 = os.path.join(tmp_path1, j)
                output_folder = os.path.join(compressed_files_folder,str(year)+'.'+ty_name)
                createfolder(output_folder)
                for k in os.listdir(tmp_path2):
                    tmp_path3 = os.path.join(tmp_path2, k)
                    for o in os.listdir(tmp_path3):
                        if dt.datetime.strptime(date_s+"."+time_s,"%Y%m%d.%H%M") <= dt.datetime.strptime(o[-16:-3],"%Y%m%d.%H%M") <= dt.datetime.strptime(date_e+"."+time_e,"%Y%m%d.%H%M"):
                            tmp_path4 = os.path.join(tmp_path3, o)
                            output_path = os.path.join(output_folder, o)

                            command = "cp {:s} {:s}".format(tmp_path4,output_path)
                            os.system(command)
                        else:
                            pass
            else:
                pass
        print("|----------------------------------------------------|")

def uncompress_and_output_numpy_files():
    # load typhoon list file
    ty_list = pd.read_excel(args.ty_list)

    compressed_files_folder = args.compressed_files_folder
    tmp_uncompressed_folder = 'tmp'
    createfolder(tmp_uncompressed_folder)

    # uncompress the file and output the readable file
    for i in sorted(os.listdir(compressed_files_folder)):
        print("-------------")
        compressed_files_folder = os.path.join(args.compressed_files_folder,i)
        numpy_files_folder = os.path.join(args.numpy_files_folder,i)
        createfolder(numpy_files_folder)
        print(i)
        for j in sorted(os.listdir(compressed_files_folder)):
            compressed_file = os.path.join(compressed_files_folder,j)
            outputtime = j[-16:-8]+j[-7:-3]
            outputtime = dt.datetime.strftime(dt.datetime.strptime(outputtime,"%Y%m%d%H%M")+dt.timedelta(hours=8),"%Y%m%d%H%M")

            if j[0] == "C":
                name = 'QPE'
                output_folder = os.path.join(numpy_files_folder,name)
                createfolder(output_folder)
            elif j[0] == "M":
                name = 'RAD'
                output_folder = os.path.join(numpy_files_folder,name)
                createfolder(output_folder)
            elif j[0] == "q":
                name = 'QPF'
                output_folder = os.path.join(numpy_files_folder,name)
                createfolder(output_folder)
            tmp_uncompressed_file = os.path.join(tmp_uncompressed_folder,name+'_'+outputtime)

            g_file = gzip.GzipFile(compressed_file)
            # 创建gzip对象
            open(tmp_uncompressed_file, "wb").write(g_file.read())
            # gzip对象用read()打开后，写入open()建立的文件中。
            g_file.close()

            tmp_file_out = os.path.join(tmp_uncompressed_folder, name+"_"+outputtime+".txt")
            bashcommand = "./fortran_codes/{:s}.out {:s} {:s}".format(name, tmp_uncompressed_file, tmp_file_out)
            os.system(bashcommand)
            data = pd.read_table(tmp_file_out,delim_whitespace=True,header=None)
            output_path = os.path.join(output_folder,i+"."+outputtime)
            print(output_path)
            np.save(output_path,data)

            os.remove(tmp_uncompressed_file)
            os.remove(tmp_file_out)

def check_data():
    # Set path
    numpy_files_folder = args.numpy_files_folder

    ty_list = pd.read_excel(args.ty_list)

    qpe_list = [x[-16:-4] for x in os.listdir(numpy_files_folder+"/"+"QPE")]
    qpf_list = [x[-16:-4] for x in os.listdir(numpy_files_folder+"/"+"QPF")]
    rad_list = [x[-16:-4] for x in os.listdir(numpy_files_folder+"/"+"RAD")]
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

    return " Check the data completely!"

if __name__ == "__main__":
    print("*" * 20)
    print("*{:^18s}*".format('Data extracter'))
    print("*" * 20)
    print("-" * 20)
    print(extract_original_data())
    uncompress_and_output_numpy_files()
