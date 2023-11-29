import torch
import torch.nn as nn
import numpy as np
import src.conv_onet.models.decoder as decoder
from torchsummary import summary
import src.config as config


def load_pretrain(nice: decoder.NICE, cfg: dict):
    ckpt = torch.load(cfg['pretrained_decoders']['coarse'], map_location=cfg['mapping']['device'])
    coarse_dict = {}
    for key, val in ckpt['model'].items():
        if ('decoder' in key) and ('encoder' not in key):
            key = key[8:]
            coarse_dict[key] = val
    nice.coarse_decoder.load_state_dict(coarse_dict)

    ckpt = torch.load(cfg['pretrained_decoders']['middle_fine'], map_location=cfg['mapping']['device'])
    middle_dict = {}
    fine_dict = {}
    for key, val in ckpt['model'].items():
        if ('decoder' in key) and ('encoder' not in key):
            if 'coarse' in key:
                key = key[8+7:]
                middle_dict[key] = val
            elif 'fine' in key:
                key = key[8+5:]
                fine_dict[key] = val
    nice.middle_decoder.load_state_dict(middle_dict)
    nice.fine_decoder.load_state_dict(fine_dict)

def load_bound(nice: decoder.NICE, cfg: dict):
    scale = cfg['scale']

    bound = torch.from_numpy(np.array(cfg['mapping']['bound'])*scale)
    bound_divisible = cfg['grid_len']['bound_divisible']
    bound[:, 1] = (((bound[:, 1]-bound[:, 0]) /
        bound_divisible).int()+1)*bound_divisible+bound[:, 0]

    nice.bound = bound
    nice.middle_decoder.bound = bound
    nice.fine_decoder.bound = bound
    nice.color_decoder.bound = bound
    nice.coarse_decoder.bound = bound * cfg['model']['coarse_bound_enlarge']


nice = decoder.NICE(coarse=True).to("cuda:0")
slim_nice = decoder.NICE(hidden_size=20, coarse=True).to("cuda:0")

cfg = config.load_config("./configs/Replica/room0.yaml", "./configs/nice_slam.yaml")

load_pretrain(nice, cfg)
load_bound(nice, cfg)

load_bound(slim_nice, cfg)


import os
from torch.optim.lr_scheduler import StepLR

EPOCH = 20
LR = 1e-4

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(slim_nice.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=250, gamma=0.7)

stages = ["coarse", "middle", "fine", "color"]

loss_record = {"coarse": [], "middle": [], "fine": [], "color": []}

for epoch in range(EPOCH):
    for stage in stages:
        optimizer.zero_grad()
        for i, pt in enumerate(os.listdir(f"./saved_inputs/{stage}")):
            data = torch.load(f"./saved_inputs/{stage}/{pt}")
            pi = data['pi'].to(cfg['mapping']['device'])
            c = data['c']
            with torch.no_grad():
                pretrained_result = nice(pi, c_grid=c, stage=stage)
            
            slim_result = slim_nice(pi, c_grid=c, stage=stage)
            loss = loss_fn(pretrained_result, slim_result) / 20
            
            loss.backward()

            if i % 20 == 0:
                optimizer.step()
                optimizer.zero_grad()
                print(f"stage: {stage}, loss: {loss.item()}")
                loss_record[stage].append(loss.item())

# Save the record
import pickle
with open("./loss_record.pkl", "wb") as f:
    pickle.dump(loss_record, f)

# Save the model
torch.save(slim_nice.state_dict(), "./slim_nice.pth") 
