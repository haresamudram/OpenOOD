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

    def get_classwise_stats(self, logits, mask):
        """
        logits: [B, P, C] tensor
        mask: [B, P] boolean or 0/1 tensor
        returns: list of per-image lists, each with tuples (class_id, frequency, mean_conf), sorted by confidence desc
        Only pixels where mask == 1 are considered.
        """
        max_vals, class_ids = logits.max(dim=2)  # [B, P], [B, P]
        B = logits.shape[0]
        results = []

        for i in range(B):
            valid_mask = mask[i].bool() 
            cls_ids = class_ids[i][valid_mask]  # Masked class IDs: [num_valid]
            confidences = max_vals[i][valid_mask].float()  # Masked confidences: [num_valid]

            if cls_ids.numel() == 0:
                results.append([])
                continue

            # Get unique class ids and inverse indices
            unique_cls, inverse_indices = torch.unique(cls_ids, return_inverse=True)
            num_classes = unique_cls.shape[0]

            # Initialize accumulators
            class_sums = torch.zeros(num_classes, device=logits.device)
            class_counts = torch.zeros(num_classes, device=logits.device)
            patch_predictions = torch.zeros(num_classes, device=logits.device) +class_ids[i,max_vals[i].argmax()]

            # Accumulate sums and counts using scatter_add
            class_sums.scatter_add_(0, inverse_indices, confidences)
            class_counts.scatter_add_(0, inverse_indices, torch.ones_like(confidences))
            
            mean_confidences = class_sums / class_counts

            # Combine into tuples
            image_result = list(zip(unique_cls.tolist(), class_counts.tolist(), mean_confidences.tolist(), patch_predictions.tolist()))

            # Sort by confidence descending
            sorted_result = sorted(image_result, key=lambda x: x[2], reverse=True)
            results.append(sorted_result)

        return results 
    
    def filter_top_class(self, results):
        filtered = []
        for image_result in results:
            frequent_prediction = sorted(image_result, key=lambda x: (-x[1], -x[2]))
            best = frequent_prediction[0][2]
            filtered.append(best)
        
        return filtered

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, flag=True):
        global_logits, local_logits, visual_logits, mask = net(data)
        global_score = torch.softmax(global_logits / self.tau, dim=-1)
        global_conf, pred = torch.max(global_score, dim=1)
        
        
        if len(local_logits.shape) > 2: 
            local_logits = local_logits.mean(dim=-1) # Local features of GalLoP

        # avg of all the patches within the mask
        local_score = torch.softmax(local_logits / self.tau, dim=-1)
        max_values, _ = torch.max(local_score, dim=-1)
        masked_logits = torch.where(mask.cuda(), max_values, torch.tensor(float('nan')))
        local_conf = torch.nanmean(masked_logits, dim=1)
        
        
        # Visual logits
        visual_prob = torch.softmax(visual_logits, dim=-1)
        visual_conf, visual_pred = torch.max(visual_prob, dim=-1)
        
        return visual_pred , visual_conf 

    def set_hyperparam(self, hyperparam: list):
        self.tau = hyperparam[0]

    def get_hyperparam(self):
        return self.tau
