from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_postprocessor import BasePostprocessor


class GLMCMPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super(GLMCMPostprocessor, self).__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.tau = self.args.tau

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        global_features, local_features = net(data)
        global_score = torch.softmax(global_features / self.tau, dim=-1)
        global_conf, pred = torch.max(global_score, dim=1)
        
        local_score = torch.softmax(local_features / self.tau, dim=-1)
        max_values, _ = torch.max(local_score, dim=2)
        local_conf, _ = torch.max(max_values, dim=1)
        
        return pred , global_conf + local_conf

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau
