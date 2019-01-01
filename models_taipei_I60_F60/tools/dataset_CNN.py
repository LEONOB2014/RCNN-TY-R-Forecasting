import numpy as np
import pandas as pd
import os
import datetime as dt
from torch.utils.data import Dataset
import torch
from .args_tools import args


class TyDataset(Dataset):
    """Typhoon dataset"""

    def __init__(self,ty_list_file,root_dir,train=True,train_num=11,test_num=2
                ,forecast_t = 1,x_tsteps=6,transform=None):
        """
        Args:
            excel_file (string): Path to the excel file with annotations.
            root_dir (string): Directory with all the files.
            events_num (int): The number of traning events.
            x_tsteps (int, 10-minutes-based): The time steps of x data. Eg: x(t),x(t-10),x(t-20)...
            forecast_t (int, n-hour-based): The n-hour ahead forecasting. Eg: y(t+1hour) = f(x(t),x(t-10),x(t-20)...)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.ty_list = pd.read_excel(ty_list_file,index_col="En name").drop("Ch name",axis=1)
        self.root_dir = root_dir
        self.x_tsteps = x_tsteps
        self.forecast_t = forecast_t
        self.transform = transform
        if train:
            self.events_num = range(0,len(self.ty_list)-test_num)
            self.events_list = self.ty_list.index[self.events_num]
        else:
            self.events_num = range(len(self.ty_list)-test_num,len(self.ty_list))
            self.events_list = self.ty_list.index[self.events_num]

        tmp = 0
        self.idx_list = pd.DataFrame([],columns=["x_start","x_end","y_start","y_end","idx_s","idx_e"])
        for i in self.events_num:
            x_s = self.ty_list.iloc[i,0]
            x_e = self.ty_list.iloc[i,1]-dt.timedelta(minutes=(self.x_tsteps-1+6*self.forecast_t)*10)
            y_s = self.ty_list.iloc[i,0]+dt.timedelta(minutes=(self.x_tsteps-1+6*self.forecast_t)*10)
            y_e = self.ty_list.iloc[i,1]

            self.files_counts = tmp + int((y_e-y_s).days*24*6 + (y_e-y_s).seconds/600)+1
            self.idx_list = self.idx_list.append({"x_start":x_s,"x_end":x_e
                                                  ,"y_start":y_s,"y_end":y_e
                                                  ,"idx_s":tmp,"idx_e":self.files_counts-1}
                                                 ,ignore_index=True)
            tmp = self.files_counts

        self.idx_list.index = self.events_list
        self.idx_list.index.name = "Typhoon"

    def __len__(self):
        return self.files_counts

    def print_idx_list(self):
        return self.idx_list

    def print_ty_list(self):
        return self.ty_list

    def __getitem__(self,idx):
        # To identify which event the idx is in.
        assert idx < self.files_counts, 'Index is out of the range of the data!'

        for i in self.idx_list.index:
            if idx > self.idx_list.loc[i,"idx_e"]:
                continue
            else:
                # determine some indexes
                idx_tmp = idx - self.idx_list.loc[i,"idx_s"]
                # print("idx_tmp: {:d} |ty: {:s}".format(idx_tmp,i))
                # print("idx: {:d} |ty: {:s}".format(idx,i))
                year = str(self.idx_list.loc[i,"x_start"].year)

                # RAD data(a tensor with shape (x_tsteps X H X W))
                rad_data = []
                for j in range(self.x_tsteps):
                    rad_files_name = dt.datetime.strftime(self.idx_list.loc[i,'x_start']+dt.timedelta(minutes=10*(idx_tmp+j))
                                                          ,format="%Y%m%d%H%M")
#                    print("rad_data: {:s}".format(year+'.'+i+"_"+rad_files_name+".npy"))
                    rad_data.append(np.load(os.path.join(self.root_dir,'RAD',year+'.'+i+"_"+rad_files_name+".npy")))
                rad_data = np.array(rad_data)

                # The endding of RAD data
                rad_e = self.idx_list.loc[i,'x_start']+dt.timedelta(minutes=10*(idx_tmp+j))

                # The begining of QPE data
                qpe_t = self.idx_list.loc[i,'y_start']+dt.timedelta(minutes=10*(idx_tmp))
                qpe_file_name = dt.datetime.strftime(qpe_t,format="%Y%m%d%H%M")
                # QPE data(a ndarray with shape (H X W))
#                print("\nqpe_data: {:s}".format(year+'.'+i+"_"+qpe_file_name+".npy"))
                qpe_data = np.load(os.path.join(self.root_dir,'QPE',year+'.'+i+"_"+qpe_file_name+".npy"))

                # return the idx of sample
                self.sample = {"RAD": rad_data, "QPE": qpe_data}
                if self.transform:
                    self.sample = self.transform(self.sample)
                return self.sample
                break

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        rad_data, qpe_data = sample['RAD'], sample['QPE']

        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'RAD': torch.from_numpy(rad_data),
                'QPE': torch.from_numpy(qpe_data)}

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    def __call__(self, sample):
        rad_data, qpe_data = sample['RAD'], sample['QPE']
        if type(self.mean) and type(self.std)== "list":
            for i in len(self.mean):
                rad_data[i] = (rad_data[i] - self.mean[i]) / self.std[i]

        # numpy data: x_tsteps X H X W
        # torch data: x_tsteps X H X W
        return {'RAD': rad_data,
                'QPE': qpe_data}

def main():
    # test dataloader
    train_dataset = TyDataset(ty_list_file="../../ty_list.xlsx",
                          root_dir="../../01_TY_database/02_wrangled_data_Taipei",
                          x_tsteps=10,
                          forecast_t = 3,
                          train=True,
                          transform = ToTensor())
    print(train_dataset[2]["RAD"].shape)

if __name__ == "__main__":
    main()
