import torch
import torch.nn.functional as F

# ------------------------------------------------------------------
def mil_bce_loss(seg_logits, labels, seg_per_video, pos_weight=None):
    B = labels.size(0)
    video_logits = seg_logits.view(B, seg_per_video).max(dim=1).values
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(pos_weight, dtype=video_logits.dtype,
                                  device=video_logits.device)
    return F.binary_cross_entropy_with_logits(video_logits, labels.float(),
                                              pos_weight=pos_weight)


# ------------------------------------------------------------------
def att_mil_bce_with_w(seg_logits, att_w, labels, seg_per_video,
                       pos_weight=None):
    B = labels.size(0)
    seg_logits = seg_logits.view(B, seg_per_video, 1)
    att_w      = att_w     .view(B, seg_per_video, 1)
    video_logits = (att_w * seg_logits).sum(1).squeeze(1)
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(pos_weight, dtype=video_logits.dtype,
                                  device=video_logits.device)
    return F.binary_cross_entropy_with_logits(video_logits, labels.float(),
                                              pos_weight=pos_weight)


# ------------------------------------------------------------------
def topk_mil_bce(seg_logits, labels, seg_per_video, k=16, pos_weight=None):
    B = labels.size(0)
    seg_logits = seg_logits.view(B, seg_per_video)          # (B,S)
    k = min(k, seg_per_video)
    topk_vals, _ = torch.topk(seg_logits, k, dim=1)         # (B,k)
    video_logits = topk_vals.mean(dim=1)                    # (B,)
    if pos_weight is not None and not torch.is_tensor(pos_weight):
        pos_weight = torch.tensor(pos_weight, dtype=video_logits.dtype,
                                  device=video_logits.device)
    return F.binary_cross_entropy_with_logits(video_logits, labels.float(),
                                              pos_weight=pos_weight)
