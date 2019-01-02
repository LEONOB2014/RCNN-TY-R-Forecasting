import sys
import gzip
from args_tools import *

def extrat_original_data():
    #load typhoon list file
    ty_list = pd.read_excel(args.ty_list)

    origianal_files_folder = args.original_files_folder
    compressed_files_folder = args.compressed_files_folder
    for i in range(len(ty_list)):
        year = ty_list["Time of issuing"][i].year
        date_s = dt.datetime.strftime(ty_list["Time of issuing"][i] - dt.timedelta(hours=8),format="%Y%m%d")
        date_e = dt.datetime.strftime(ty_list["Time of canceling"][i] - dt.timedelta(hours=8),format="%Y%m%d")
        time_s = dt.datetime.strftime(ty_list["Time of issuing"][i] - dt.timedelta(hours=8),format="%H%M")
        time_e = dt.datetime.strftime(ty_list["Time of canceling"][i] - dt.timedelta(hours=8),format="%H%M")
        ty_name = ty_list["En name"][i]

        print("|{:8s}| start time: {:s} | end time: {:s} |".format(ty_name+"",date_s,date_e))
        print("|{:8s}|             {:8s} |           {:8s} |".format(" ",time_s,time_e))

        tmp_path1 = os.path.join(origianal_files_folder,str(year))
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

def uncompress_and_output_readable_files():
    # load typhoon list file
    ty_list = pd.read_excel(args.ty_list)

    compressed_files_folder = args.compressed_files_folder
    tmp_uncompressed_folder = 'tmp'
    createfolder(tmp_uncompressed_folder)

    # uncompress the file and output the readable file
    for i in sorted(os.listdir(compressed_files_folder)):
        print("-------------")
        compressed_files_folder = args.compressed_files_folder+"/"+i
        readable_files_folder = args.readable_files_folder+"/"+i
        createfolder(readable_files_folder)
        print(i)
        for j in sorted(os.listdir(compressed_files_folder)):
            compressed_file = compressed_files_folder+'/'+j
            outputtime = j[-16:-8]+j[-7:-3]
            outputtime = dt.datetime.strftime(dt.datetime.strptime(outputtime,"%Y%m%d%H%M")+dt.timedelta(hours=8),"%Y%m%d%H%M")

            if j[0] == "C":
                name = 'QPE'
            elif j[0] == "M":
                name = 'RAD'
            elif j[0] == "q":
                name = 'QPF'
            tmp_uncompressed_file = tmp_uncompressed_folder+'/'+name+'_'+outputtime

            g_file = gzip.GzipFile(compressed_file)
            # 创建gzip对象
            open(tmp_uncompressed_file, "wb").write(g_file.read())
            # gzip对象用read()打开后，写入open()建立的文件中。
            g_file.close()

            file_out = readable_files_folder+"/"+name+"_"+outputtime+".txt"

            bashcommand = "./../{:s}.out {:s} {:s}".format(name, tmp_uncompressed_file, file_out)

            os.system(bashcommand)
            os.remove(tmp_uncompressed_file)

if __name__ == "__main__":
    extrat_original_data()
    uncompress_and_output_readable_files()
