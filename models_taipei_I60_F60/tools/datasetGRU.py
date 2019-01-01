import os
import numpy as np
import pandas as pd
import datetime as dt
# import torch module
import torch
from torch.utils.data import Dataset


class TyDataset(Dataset):
    """Typhoon dataset"""

    def __init__(self, ty_list_file, root_dir, train=True, train_ratio=0.75,
                input_frames=5, output_frames=18, transform=None):
        """
        Args:
            ty_list_file (string): Path of the typhoon list file.
            root_dir (string): Directory with all data.
            train (boolean): Extract training set or not.
            train_num (int): The number of training events.
            test_num (int): The number of training events.
            input_frames (int, 10-minutes-based): The frames of input data.
            output_frames (int, 10-minutes-based): The frames of output data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        super().__init__()
        self.ty_list = pd.read_excel(ty_list_file, index_col="En name").drop("Ch name",axis=1)
        self.root_dir = root_dir
        self.input_frames = input_frames
        self.output_frames = output_frames
        self.transform = transform
        self.train_num = int(len(self.ty_list)*(train_ratio))

        if train:
            self.events_num = range(0, self.train_num)
            self.events_list = self.ty_list.index[self.events_num]
        else:
            self.events_num = range(self.train_num, len(self.ty_list))
            self.events_list = self.ty_list.index[self.events_num]

        tmp = 0
        # get the indexes of the data
        self.idx_list = pd.DataFrame([],columns=["frame_start","frame_end","idx_s","idx_e"])
        for i in self.events_num:
            frame_s = self.ty_list.iloc[i,0]
            frame_e = self.ty_list.iloc[i,1]-dt.timedelta(minutes=(input_frames+output_frames-1)*10)

            self.total_frames = tmp + int((frame_e-frame_s).days*24*6 + (frame_e-frame_s).seconds/600)+1
            self.idx_list = self.idx_list.append({"frame_start":frame_s,"frame_end":frame_e
                                                  ,"idx_s":tmp,"idx_e":self.total_frames-1}
                                                 ,ignore_index=True)
            tmp = self.total_frames

        self.idx_list.index = self.events_list
        self.idx_list.index.name = "Typhoon"

    def __len__(self):
        return self.total_frames

    def return_train_num(self):
        return self.train_num

    def return_idx_list(self):
        return self.idx_list

    def return_ty_list(self):
        return self.ty_list

    def __getitem__(self,idx):
        # To identify which event the idx is in.
        assert idx < self.total_frames, 'Index is out of the range of the data!'

        for i in self.idx_list.index:
            if idx > self.idx_list.loc[i,"idx_e"]:
                continue
            else:
                # determine some indexes
                idx_tmp = idx - self.idx_list.loc[i,"idx_s"]
                # print("idx_tmp: {:d} |ty: {:s}".format(idx_tmp,i))
                # print("idx: {:d} |ty: {:s}".format(idx,i))
                year = str(self.idx_list.loc[i,'frame_start'].year)

                # RAD data(a tensor with shape (input_frames X H X W))
                rad_data = []
                for j in range(self.input_frames):
                    rad_file_time = dt.datetime.strftime(self.idx_list.loc[i,'frame_start']+dt.timedelta(minutes=10*(idx_tmp+j))
                                                          ,format="%Y%m%d%H%M")
                    # print("rad_data: {:s}".format(year+'.'+i+"_"+rad_file_time+".npy"))
                    rad_data.append(np.expand_dims(np.load(os.path.join(self.root_dir,'RAD',year+'.'+i+"_"+rad_file_time+".npy")), axis=0))
                rad_data = np.array(rad_data)

                # QPE data(a tensor with shape (output_frames X H X W))
                qpe_data = []

                for j in range(self.input_frames,self.input_frames+self.output_frames):
                    qpe_file_time = dt.datetime.strftime(self.idx_list.loc[i,'frame_start']+dt.timedelta(minutes=10*(idx_tmp+j))
                                                          ,format="%Y%m%d%H%M")
                    # print("qpe_data: {:s}".format(year+'.'+i+"_"+qpe_file_time+".npy"))
                    qpe_data.append(np.load(os.path.join(self.root_dir,'QPE',year+'.'+i+"_"+qpe_file_time+".npy")))
                qpe_data = np.array(qpe_data)
                # return the idx of sample
                self.sample = {"RAD": rad_data, "QPE": qpe_data}
                if self.transform:
                    self.sample = self.transform(self.sample)
                return self.sample


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
        return {'RAD': rad_data, 'QPE': qpe_data}



if __name__ == "__main__":
    # test TyDataset
    train_dataset = TyDataset(ty_list_file="../../../ty_list.xlsx",
                          root_dir="../../../01_Radar_data/02_wrangled_files_Taipei_I.60_F.60",
                          input_frames=5,
                          output_frames=20,
                          train=True,
                          transform = ToTensor())
    print("QPE size: ",train_dataset[2]["QPE"].shape)
    print("RAD size: ",train_dataset[2]["RAD"].shape)
