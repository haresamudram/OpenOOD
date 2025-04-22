from typing import Any
from tqdm import tqdm
import json, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import openood.utils.comm as comm
import pandas as pd


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loader: DataLoader,
                  progress: bool = True):
        pred_list, conf_list, label_list, image_name = [], [], [], []
        for batch in tqdm(data_loader,
                          disable=not progress or not comm.is_main_process()):
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            # if 23 in label.cpu():
            #     print('Batch with bald eagle')
            # if 352 in label.cpu():
            #     print('Batch with hartebeest')
            # if 353 in label.cpu():
            #     print('Batch with impala')
            pred, conf = self.postprocess(net, data, flag=False)

            pred_list.append(pred.cpu())
            conf_list.append(conf.cpu())
            label_list.append(label.cpu())
            #image_name.append(batch['image_name'])


        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)
        #image_name = [item for sublist in image_name for item in sublist]

        # Creating DataFrame
        # df = pd.DataFrame({
        #     'pred': pred_list,
        #     'conf': conf_list,
        #     'label': label_list,
        #     'image_path': image_name
        # })

        # Saving to CSV
        #df.to_csv('/ood_datadrive/ood/ood_inferencing/inaturalist_key_phrases.csv', index=False)

       
        return pred_list, conf_list, label_list
