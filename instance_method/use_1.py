import torch
from utils.use import *

class USE():
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

        self.pre_adaptation(args)

        # 1. Initialize tracking variables for the first step
        tracked_pseudo_label = None
        tracked_ent = None
        tracked_alpha = None
        tracked_idx = None
        weak_pred_zs = None

        for j in range(args.tta_steps):
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)

            pred_zs = outputs[0].argmax()
            pred_augs = outputs[1:].argmax(dim=1)
            
            # 2. Capture the entropy (ent) instead of discarding it with '_'
            pred_mean, ent, idx, alpha = select_and_cal_ent(
                outputs,
                args.selection_p,
                alpha=args.alpha,
                scale=args.scale,
            )

            # 3. Store metrics on the first iteration BEFORE weights update
            if j == 0:
                tracked_pseudo_label = pred_mean
                tracked_ent = ent
                tracked_idx = idx
                tracked_alpha = alpha
                weak_pred_zs = pred_zs.item()

            if not (pred_augs[idx] != pred_zs).sum().item() == 0:
                with torch.amp.autocast('cuda'):
                    loss, pred_ = rce_loss(outputs[0], outputs[1:][idx], alpha)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        tta_outputs = self.model(images)

                final_pred, _ = cal_ent(tta_outputs[0], tta_outputs[1:][idx], alpha)

            else:
                final_pred = pred_zs.item()

        # 4. Return the final adapted prediction alongside the first-step tracking metrics
        return final_pred, tracked_pseudo_label, tracked_ent, tracked_idx, tracked_alpha, weak_pred_zs
