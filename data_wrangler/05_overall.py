from args_tools import *

def overall():
    # Taipei
    study_area = args.study_area
    # Set path
    wrangled_files_folder = args.wrangled_files_folder+"_"+study_area
    dropbox_files_folder = args.dropbox_folder

    file_out = open("../overall.txt","w")
    dropbox_file_out = open(dropbox_files_folder+"/overall.txt","w")
    dropbox_file_out_mu_std = open(dropbox_files_folder+"/mu_std.txt","w")
    for i in sorted(os.listdir(wrangled_files_folder)):
        tmp_path = wrangled_files_folder+"/"+i
        tmp=0
        tmp_max = []
        tmp_min = []
        tmp_max_file = []
        tmp_min_file = []
        tmp_ty = []
        mu = 0
        std = 0
        for j in sorted(os.listdir(tmp_path)):
            if j[:-17] not in tmp_ty:
                tmp_ty.append(j[:-17])
                tmp_max.append(0)
                tmp_min.append(100)
                tmp_max_file.append(0)
                tmp_min_file.append(0)
                tmp = tmp+1

            file_in = tmp_path+"/"+j
            data = np.load(file_in)
            mu += np.sum(data)
            if tmp_max[tmp-1] < np.max(data):
                tmp_max[tmp-1] = np.max(data)
                tmp_max_file[tmp-1] = j
            if tmp_min[tmp-1] > np.min(data):
                tmp_min[tmp-1] = np.min(data)
                tmp_min_file[tmp-1] = j

        mu = mu/(len(os.listdir(tmp_path))*data.size)
        for j in sorted(os.listdir(tmp_path)):
            file_in = tmp_path+"/"+j
            data = np.load(file_in)
            std += np.sum((data-mu)**2)
        std = np.sqrt(std/(len(os.listdir(tmp_path))*data.size))
        dropbox_file_out_mu_std.writelines("{:>5s}: |mu: {:6.3f} |std: {:6.3f}\n".format(i,mu,std))

        dropbox_file_out.writelines("-----------------------------------------------------------------------------------------------------------------------------\n")
        dropbox_file_out.writelines(i+"\n")

        file_out.writelines("-----------------------------------------------------------------------------------------------------------------------------\n")
        file_out.writelines(i+"\n")
        for i in np.arange(len(tmp_ty)):
            dropbox_file_out.writelines("{:s}\t|min:{:8.2f} file_min:{:s}\t|max:{:8.2f} file_max:{:s}\n".format(tmp_ty[i],tmp_min[i],tmp_min_file[i],tmp_max[i],tmp_min_file[i]))
            file_out.writelines("{:s}\t|min:{:8.2f} file_min:{:s}\t|max:{:8.2f} file_max:{:s}\n".format(tmp_ty[i],tmp_min[i],tmp_min_file[i],tmp_max[i],tmp_min_file[i]))
    dropbox_file_out.close()
    file_out.close()



if __name__ == "__main__":
    overall()
