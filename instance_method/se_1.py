import torch
import torch.nn.functional as F
import numpy as np

def compute_jsd(p, q):
    """Computes Jensen-Shannon Divergence between two probability distributions."""
    m = 0.5 * (p + q)
    # F.kl_div expects the first input in log-space and target in prob-space
    kl_p = F.kl_div(m.log(), p, reduction='sum')
    kl_q = F.kl_div(m.log(), q, reduction='sum')
    return 0.5 * (kl_p + kl_q)

def select_and_cal_ent(outputs, selection_p=0.1, alpha=0, tau=1.0):
    # outputs[0] is weak augmentation (p_w)
    # outputs[1:] are strong augmentations
    
    # 1. Convert logits to probabilities
    p_w = outputs[0].softmax(dim=0)
    p_s_all = outputs[1:].softmax(dim=1)
    
    # 2. Select top confident strong augmentations (same as USE filter)
    batch_entropy = -(p_s_all * outputs[1:].log_softmax(1)).sum(1)
    selected_num = max(1, int(batch_entropy.size(0) * selection_p))
    idx = torch.argsort(batch_entropy, descending=False)[:selected_num]
    
    # Calculate mean strong prediction over selected views
    p_s_mean = p_s_all[idx].mean(dim=0)
    
    # 3. Calculate Entropies
    H_w = -(p_w * outputs[0].log_softmax(0)).sum()
    H_s = -(p_s_mean * torch.log(p_s_mean + 1e-8)).sum()
    
    # 4. Calculate JSD Penalty
    jsd_penalty = compute_jsd(p_w, p_s_mean)
    
    # 5. Continuous Agreement-Aware Weighting (replaces Eq 6)
    w_w = torch.exp(-H_w / tau) * (1.0 - jsd_penalty)
    w_s = torch.exp(-H_s / tau)
    
    beta_star = w_w / (w_w + w_s + 1e-8)
    
    # 6. Final Pseudo-Label & Ensemble Entropy
    q_x = beta_star * p_w + (1.0 - beta_star) * p_s_mean
    pred_mean = q_x.argmax().item()
    
    output_clamp = torch.clamp(q_x, min=1e-7)
    ent_ = -(output_clamp * output_clamp.log()).sum() / np.log(len(output_clamp))
    
    return pred_mean, ent_.item(), idx, beta_star.item()

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
        
        pred_mean, _, _, alpha = select_and_cal_ent(
                output, 0.1, args.alpha, args.scale
            )

        return pred_mean

