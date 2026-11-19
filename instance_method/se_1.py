import torch
import numpy as np

import torch.nn.functional as F

def cal_ent(zs, ens, alpha):
    ens_pred = alpha * zs.softmax(dim=0) + (1.0 - alpha) * ens.softmax(dim=1).mean(dim=0)
    pred_ = ens_pred.argmax()

    output_ = torch.clamp(ens_pred, min=1e-7)
    ent_ = -(output_ * output_.log()).sum() / np.log(len(output_))

    return pred_.item(), ent_.item(), output_.detach()

def self_reward_split(strong_logits, q):
  with torch.no_grad():
        y_q = q.argmax()                                  # 12a
        pred = strong_logits.argmax(dim=1)
        r = 2 * (pred == y_q).long() - 1                  # 12b
        agree = r == 1                                    # 12c
  return {'y_q': y_q, 'pred': pred, 'r': r, 'agree': agree}
def select_and_cal_ent(outputs, selection_p, alpha=None, scale=0.4,sel_num =6 ):
   
    batch_entropy = -(outputs[1:].softmax(1) * outputs[1:].log_softmax(1)).sum(1)
    zs_entropy = -(outputs[0].softmax(0) * outputs[0].log_softmax(0)).sum()
    
    # 选择最低熵的样本
    # selected_num = max(1, int(batch_entropy.size(0) * selection_p))
    selected_num =6
    idx = torch.argsort(batch_entropy, descending=False)[:selected_num]

    # print(selected_num)
    
    # 计算alpha
    if alpha is not None:
        final_alpha = alpha
    else:
        rank_ = (zs_entropy < batch_entropy).sum().item() / len(batch_entropy)
        diff=rank_-0.5
        sign = 1 if diff > 0 else (-1 if diff < 0 else 0)
        final_alpha = sign*diff*diff * scale + 0.5
    
    # 计算cal_ent
    pred_mean, ent,pseudolabel = cal_ent(outputs[0], outputs[1:][idx], final_alpha)
    
    return pred_mean, ent, idx, final_alpha,pseudolabel

def se_stage(outputs, selection_p, alpha=None, scale=0.4, sel_num=6):
    pred, ent, idx, a, q = select_and_cal_ent(outputs, selection_p, alpha, scale, sel_num)
    return pred, ent, idx, a, q, self_reward_split(outputs[1:], q)

class SE():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def pre_adaptation(self, args):
        if args.load is not None:
            self.model.prompt_learner.ctx_init_state = args.pretrained_ctx
        with torch.no_grad():
            self.model.reset()
        self.optimizer = torch.optim.AdamW(self.model.prompt_learner.parameters(), args.lr)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1000)

    def adaptation_process(self, image, images, args):
        assert args.tta_steps > 0

        # self.pre_adaptation(args)
        selected_idx = None
        pred_dict = []
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                output = self.model(images)
        
        pred, ent, idx, alpha, q, rw = se_stage(output, 0.1, args.alpha, args.scale)
        info = {'conf': q[pred].item(), 'queried': False, 'updated': False}
        return pred, ent, q, alpha, idx, info

