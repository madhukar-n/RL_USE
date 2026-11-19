import torch

from utils.use import *


class USE():
    def __init__(self, model, device):
        self.model = model
        self.device = device
        # Number of samples for which binary feedback was requested.
        self.num_feedback_queries = 0

    def pre_adaptation(self, args):
        if args.load is not None:
            self.model.prompt_learner.ctx_init_state = args.pretrained_ctx
        with torch.no_grad():
            self.model.reset()
        self.optimizer = torch.optim.AdamW(self.model.prompt_learner.parameters(), args.lr)
        self.scaler = torch.amp.GradScaler('cuda', init_scale=1000)

    def adaptation_process(self, image, images, args, target=None):
        """Adapt the prompt on one test sample and return (final_pred, first_step_entropy).

        With --use_feedback, a sample whose first-step ensemble entropy exceeds
        --feedback_ent_threshold is treated as low confidence: one binary feedback
        signal (is the pseudo-label correct?) is queried from `target`, and the prompt
        is updated with a REINFORCE loss (reward +1 / -1) instead of reverse cross-entropy.
        High-confidence samples use the standard USE update.
        """
        assert args.tta_steps > 0

        self.pre_adaptation(args)

        first_ent = None
        feedback = None  # (label, reward) once queried; reused for later TTA steps

        for j in range(args.tta_steps):
            with torch.amp.autocast('cuda'):
                outputs = self.model(images)

            pred_zs = outputs[0].argmax()
            pred_augs = outputs[1:].argmax(dim=1)
            pred_mean, ent, idx, alpha = select_and_cal_ent(
                outputs,
                args.selection_p,
                alpha=args.alpha,
                scale=args.scale,
            )

            if j == 0:
                first_ent = ent
                if (getattr(args, 'use_feedback', False) and target is not None
                        and ent > args.feedback_ent_threshold):
                    # One binary query per sample: is the first-step pseudo-label correct?
                    reward = 1.0 if int(pred_mean) == int(target) else -1.0
                    feedback = (pred_mean, reward)
                    self.num_feedback_queries += 1

            if not (pred_augs[idx] != pred_zs).sum().item() == 0:

                with torch.amp.autocast('cuda'):
                    if feedback is not None:
                        label, reward = feedback
                        log_prob = outputs[0].float().log_softmax(dim=0)[label]
                        loss = -reward * log_prob
                    else:
                        loss, _ = rce_loss(outputs[0], outputs[1:][idx], alpha)

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

        return final_pred, first_ent
