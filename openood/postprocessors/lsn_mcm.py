from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor

class LSNPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(LSNPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, [outputs_yes, outputs_no1, outputs_no2, outputs_no3] = net(data)
    
        similarity_yes = F.softmax(outputs_yes, dim=1)
        similarity_no1 = F.softmax((1-outputs_no1)/5 , dim=1)
        similarity_no2 = F.softmax((1-outputs_no2)/5 , dim=1)
        similarity_no3 = F.softmax((1-outputs_no3)/5 , dim=1)
        
        max_out_yes, index_out_yes = torch.max(similarity_yes,dim = 1)
        max_out_no1, index_out_no = torch.max(similarity_no1,dim = 1)
        max_out_no2, index_out_no = torch.max(similarity_no2,dim = 1)
        max_out_no3, index_out_no = torch.max(similarity_no3,dim = 1)
        max_out_no = (max_out_no1 + max_out_no2 + max_out_no3)/3
        max_out_yes_plus = max_out_yes + max_out_no
        
        return index_out_yes, max_out_yes_plus

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau