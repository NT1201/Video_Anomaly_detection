import torch, numpy as np
def false_alarm_rate(model, loader, device, thr):
    total, fp = 0, 0
    model.eval()
    with torch.no_grad():
        for v in loader:                                   # normal-only
            s = torch.sigmoid(model(v.view(-1, v.size(-1)).to(device)))
            s = s.cpu().numpy().ravel()
            fp  += (s >= thr).sum(); total += s.size
    return 100 * fp / (total + 1e-8)
