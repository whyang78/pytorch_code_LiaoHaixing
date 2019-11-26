import h5py
import torch
from torch.utils.data import Dataset

class h5_dataset(Dataset):
    def __init__(self,h5File_list):
        label_file=h5py.File(h5File_list[0],'r')
        self.label=torch.from_numpy(label_file['label'].value)
        self.nsample=self.label.size(0)

        temp_feature=torch.FloatTensor()
        for h in h5File_list:
            data_file=h5py.File(h,'r')
            feature=torch.from_numpy(data_file['data'].value)
            temp_feature=torch.cat((temp_feature,feature),dim=1) #按特征维度进行拼接
        self.feature=temp_feature

    def __getitem__(self, index):
        assert index < len(self)
        return (self.feature[index],self.label[index])

    def __len__(self):
        return self.nsample