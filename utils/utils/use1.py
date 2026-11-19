import torch
import numpy as np
# 1) cal_ent: return q(x), detached
def cal_ent(zs, ens, alpha):
    ens_pred = alpha * zs.softmax(dim=0) + (1.0 - alpha) * ens.softmax(dim=1).mean(dim=0)
    pred_ = ens_pred.argmax()
    output_ = torch.clamp(ens_pred, min=1e-7)
    ent_ = -(output_ * output_.log()).sum() / np.log(len(output_))
    return pred_.item(), ent_.item(), output_.detach()

# 2) select_and_cal_ent: add sel_num, return q
def select_and_cal_ent(outputs, selection_p, alpha=None, scale=0.4, sel_num=6):
    with torch.no_grad():
        batch_entropy = -(outputs[1:].softmax(1) * outputs[1:].log_softmax(1)).sum(1)
        zs_entropy = -(outputs[0].softmax(0) * outputs[0].log_softmax(0)).sum()
        selected_num = max(1, min(sel_num, outputs.shape[0] - 1))
        idx = torch.argsort(batch_entropy, descending=False)[:selected_num]
        if alpha is not None:
            final_alpha = alpha
        else:
            rank_ = (zs_entropy < batch_entropy).sum().item() / len(batch_entropy)
            diff = rank_ - 0.5
            sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
            final_alpha = sign * diff * diff * scale + 0.5
        pred_mean, ent, q = cal_ent(outputs[0], outputs[1:][idx], final_alpha)
    return pred_mean, ent, idx, final_alpha, q

# 3) SE stage: steps 12a-12c
def self_reward_split(strong_logits, q):
    with torch.no_grad():
        y_q = q.argmax()                                  # 12a
        pred = strong_logits.argmax(dim=1)
        r = 2 * (pred == y_q).long() - 1                  # 12b
        agree = r == 1                                    # 12c
    return {'y_q': y_q, 'pred': pred, 'r': r, 'agree': agree}

def se_stage(outputs, selection_p, alpha=None, scale=0.4, sel_num=6):
    pred, ent, idx, a, q = select_and_cal_ent(outputs, selection_p, alpha, scale, sel_num)
    return pred, ent, idx, a, q, self_reward_split(outputs[1:], q)

# 4) USE stage: step 14a
def self_reward_loss(view_logits, q, rw, lam=0.5, eps=1e-6, weight=None):
    view_logits = view_logits.float()
    q = q.detach()
    y_q, pred, agree = rw['y_q'], rw['pred'], rw['agree']
    logp = view_logits.log_softmax(dim=1)
    loss = view_logits.new_zeros(())
    if agree.any():
        loss = loss - logp[agree, y_q].mean()
    if (~agree).any():
        p_wrong = logp[~agree].gather(1, pred[~agree].unsqueeze(1)).exp().squeeze(1)
        loss = loss - lam * torch.log(1.0 - p_wrong + eps).mean()
    return (q.max() if weight is None else weight) * loss

# 5) rce_loss: only if you don't already have it (my reconstruction of Eq. 8)
def rce_loss(zs, ens, alpha):
    pw, ps = zs.float().softmax(0), ens.float().softmax(1)
    q = (alpha * pw + (1 - alpha) * ps.mean(0)).detach()
    p_bar = torch.cat([pw.unsqueeze(0), ps], 0).mean(0)
    return -(p_bar * q.clamp(min=1e-7).log()).sum(), q.argmax().item()