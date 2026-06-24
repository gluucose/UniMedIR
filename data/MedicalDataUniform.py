import torch
from torch.utils.data import Dataset
import os
import numpy as np
from .common import dataIO, transformData
import glob
import json
import random

io=dataIO() 
transform = transformData()

class Train_Data(Dataset):
    def __init__(self, root_dir, modality_list = ["PET"], patch_size=128):

        
        
        
        self.LQ_paths = [] 
        self.HQ_paths = []

        desc_path = os.path.join(root_dir, "descriptions.json")  ### text prompt path
        with open(desc_path, "r") as f:
            self.prompt_data = json.load(f)
        
        for modality in modality_list:
            tmp_paths = glob.glob(os.path.join(root_dir, modality, "train", "LQ", "*.bin"))
            
            for p in tmp_paths:  
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))

        self.length = len(self.LQ_paths) 
        self.label_dict = {
            "CT": 0,
            "PET": 1,
            "MRI": 2,
            "CT2": 3,
            "CT3": 4,
            "PET2": 5,
            "WSI": 6,
            } 
        self.patch_size = patch_size


    def __len__(self):
        return self.length

    def analyze_path(self, path): 
        # path_parts = path.split('/')
        from pathlib import Path
        path_parts = list(Path(path).parts)
        
        file_name = path_parts[-1] 
        base_name, _ = os.path.splitext(file_name) 
        
        modality = path_parts[-4] 
        return modality, base_name

    def get_prompt(self, modality):   
        if modality not in self.prompt_data:
            return ""

        # get text prompts in the prompt set
        candidates = [self.prompt_data[modality]["standard_prompt"]] + self.prompt_data[modality]["augmented_prompts"]
        prompt = random.choice(candidates)

        return prompt


    def __getitem__(self, idx):

       
        imgLQ = io.load(self.LQ_paths[idx]) 
        imgHQ =io.load(self.HQ_paths[idx])
        # print('self.LQ_paths[idx]', self.LQ_paths[idx])
        # print('imageLQ', imgLQ.shape, 'imgHQ', imgHQ.shape)
        
        # import pdb 
        # pdb.set_trace() 
        modality, _ = self.analyze_path(self.LQ_paths[idx]) 

        imgLQ = torch.tensor(transform.normalize(imgLQ, modality))
        imgHQ = torch.tensor(transform.normalize(imgHQ, modality))
        # print('imageLQ', imgLQ.shape, 'imgHQ', imgHQ.shape)

        cat_pic = torch.cat([imgLQ, imgHQ], dim=0).unsqueeze(1)
        cat_pic = transform.random_crop(tensor = cat_pic, patch_size=[self.patch_size, self.patch_size]).squeeze(1)
        imgLQ, imgHQ = torch.chunk(cat_pic, 2, dim=0) 
        
        class_label = self.label_dict[modality]

        prompt = self.get_prompt(modality)
        
        return imgLQ, imgHQ, class_label, prompt


class Test_Data(Dataset):
    def __init__(self, root_dir, modality_list = ["PET2", "PET", "CT", "CT2", "MRI"], use_num = None, target_folder="validation"):
        
        self.LQ_paths = [] 
        self.HQ_paths = []

        desc_path = os.path.join(root_dir, "descriptions.json")
        with open(desc_path, "r") as f:
            self.prompt_data = json.load(f)
        
        for modality in modality_list: 
            tmp_paths = glob.glob(os.path.join(root_dir, modality, target_folder, "LQ", "*.nii")) 
            
            use_num = len(tmp_paths) if use_num is None else use_num
            
            for num in range(use_num):
                p = tmp_paths[num]
                self.LQ_paths.append(p)
                self.HQ_paths.append(p.replace("LQ", "HQ"))  

        self.length = len(self.LQ_paths) 

    def analyze_path(self, path):
        # path_parts = path.split('/')
        from pathlib import Path
        path_parts = Path(path).parts
        
        file_name = path_parts[-1] 
        base_name, _ = os.path.splitext(file_name) 

        # print('modality', path_parts)
        modality = path_parts[-4]

        return modality, base_name


    def get_prompt(self, modality):
        if modality not in self.prompt_data:
            return ""

        prompt = self.prompt_data[modality]["standard_prompt"]

        return prompt
        

    def __len__(self):
        return self.length 

    def __getitem__(self, idx):

       
        imgLQ = io.load(self.LQ_paths[idx])
        imgHQ =io.load(self.HQ_paths[idx]) 
        
        modality, file_name = self.analyze_path(self.LQ_paths[idx]) 
        
        # import pdb 
        # pdb.set_trace()

        
        imgLQ = transform.normalize(imgLQ, modality)
        imgHQ = transform.normalize(imgHQ, modality)
        
        imgLQ = torch.from_numpy(imgLQ).unsqueeze(0) 
        imgHQ = torch.from_numpy(imgHQ).unsqueeze(0)

        prompt = self.get_prompt(modality)

        return imgLQ, imgHQ, modality, file_name, prompt








class DataSampler:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.data_iter)
        except StopIteration:
            # 如果 DataLoader 中的数据采样完了，重新 shuffle 数据

            self.data_iter = iter(self.dataloader)
            batch = next(self.data_iter)

        return batch

# dataset = Train_Data() 
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True,drop_last=True) 
# data_sampler = DataSampler(data_loader) 

if __name__ == "__main__": 
    from tqdm import tqdm 
    from torch.utils.data import DataLoader
    
    data_root = "../dataset/All-in-One"
    modality_list = ["PET", "CT", "MRI"]
    train_loader_list = []


    
    dataset = {
        'train': Train_Data(root_dir = data_root, modality_list=["PET", "MRI"]),
        # 'val': Test_Data(root_dir = data_root, modality_list = ["MRI"], use_num=4), 
        # 'test': Test_Data(root_dir=data_root, center_name="m660-1", use_num=-1),
        
        } 
    train_loader = DataLoader(dataset['train'], batch_size=1, shuffle=False) 
    
    print("length:", len(train_loader))
    
    # valid_loader = DataLoader(Val_Data(root=data_root, center_list=center_list), batch_size=1) 
    
    # test_loader = DataLoader(Test_Data(root=data_root, center_list=center_list), batch_size=1)
    
    

    for counter, data in enumerate(tqdm(train_loader)): 
        # import pdb
        # pdb.set_trace()
        print('counter', counter, 'data', data[0].shape)
