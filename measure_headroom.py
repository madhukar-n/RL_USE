import argparse
import csv
import json
import os
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from clip.custom_clip import get_coop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from data.cls_to_names import *
from data.imagenet_variants import imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
from instance_method.headroom import Headroom
from utils.tools import set_random_seed


def get_classnames(dset):
    if len(dset) > 1:
        return eval("{}_classes".format(dset.lower()))
    assert dset in ['A', 'R', 'K', 'V', 'I']
    classnames_all = imagenet_classes
    if dset in ['A', 'R', 'V']:
        label_mask = eval("imagenet_{}_mask".format(dset.lower()))
        if dset == 'R':
            return [classnames_all[i] for i, m in enumerate(label_mask) if m]
        return [classnames_all[i] for i in label_mask]
    return classnames_all


def format_markdown_summary_table(results_by_dset, arch):
    lines = [
        f"# Step 0 Headroom Summary (`{arch}`)",
        "",
        "| Dataset | N | 1. SE $q$ (USE) | 1b. SE $q$ (Quad) | 2a. Oracle $q$ (SE-Mixed) | 2b. Oracle $q$ (Pure Aug) | 2c. Oracle $q$ (Pure All) | 3a. ZS View 0 | 3b. Min-Ent Single Aug | 3c. Best Fixed View | 3d. Oracle Single View (All) | Headroom (Mixed - SE) | Headroom (Pure - SE) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dset, s in results_by_dset.items():
        lines.append(
            f"| {dset} | {s['num_samples']} "
            f"| {s['acc_q_se_use']:.2f}% "
            f"| {s['acc_q_se_quad']:.2f}% "
            f"| {s['acc_q_oracle_se_mixed']:.2f}% "
            f"| {s['acc_q_oracle_pure_aug']:.2f}% "
            f"| {s['acc_q_oracle_pure_all']:.2f}% "
            f"| {s['acc_zs_view0']:.2f}% "
            f"| {s['acc_min_ent_single_aug']:.2f}% "
            f"| {s['acc_best_fixed_view_all']:.2f}% "
            f"| {s['acc_oracle_single_all']:.2f}% "
            f"| +{s['headroom_oracle_se_mixed_over_se']:.2f}% "
            f"| +{s['headroom_oracle_pure_all_over_se']:.2f}% |"
        )
    return "\n".join(lines)


def evaluate_dataset(dset, args, normalize, out_file):
    set_random_seed(args.seed)

    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution),
    ])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    data_transform = AugMixAugmenter(
        base_transform,
        preprocess,
        n_views=args.batch_size - 1,
        augmix=len(dset) > 1,
    )

    val_dataset = build_dataset(dset, data_transform, args.data, mode=args.dataset_mode)
    if args.max_samples is not None:
        val_dataset = torch.utils.data.Subset(
            val_dataset, range(min(args.max_samples, len(val_dataset)))
        )

    header_msg = f"\n>>> Evaluating dataset [{dset}] (samples={len(val_dataset)}, arch={args.arch})"
    print(header_msg)
    out_file.write(header_msg + "\n")
    out_file.flush()

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
    )

    classnames = get_classnames(dset)
    model = get_coop(args.arch, classnames, args.gpu, args.n_ctx, args.ctx_init)

    if args.load is not None:
        load_rel = {
            'RN50': 'rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50',
            'ViT-B/16': 'vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed2/prompt_learner/model.pth.tar-50',
        }[args.arch]
        load_path = os.path.join(args.load, load_rel)
        pretrained_ctx = torch.load(load_path, weights_only=False, map_location='cpu')['state_dict']['ctx']
        assert pretrained_ctx.size()[0] == args.n_ctx
        model.prompt_learner.ctx_init_state = pretrained_ctx
        with torch.no_grad():
            model.reset()

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    evaluator = Headroom(model, args.gpu)
    evaluator.model.eval()

    for i, data in enumerate(val_loader):
        target = data[1]
        images = data[0]
        if isinstance(images, list):
            if torch.cuda.is_available():
                images = [img.cuda(args.gpu, non_blocking=True) for img in images]
        else:
            if len(images.size()) > 4:
                assert images.size()[0] == 1
                images = images.squeeze(0)
            if torch.cuda.is_available():
                images = images.cuda(args.gpu, non_blocking=True)

        if isinstance(images, list):
            images = torch.cat(images, dim=0)

        evaluator.evaluate_sample(images, target, args)

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(val_loader):
            s = evaluator.summary()
            log_line = (
                f"[{dset}] iter:{i + 1}/{len(val_loader)} | "
                f"SE_q={s['acc_q_se_use']:.2f}% (quad={s['acc_q_se_quad']:.2f}%) | "
                f"Oracle_q(mixed)={s['acc_q_oracle_se_mixed']:.2f}% | "
                f"Oracle_q(pure)={s['acc_q_oracle_pure_all']:.2f}% | "
                f"ZS_view0={s['acc_zs_view0']:.2f}% | "
                f"MinEnt_view={s['acc_min_ent_single_aug']:.2f}%"
            )
            print(log_line)
            out_file.write(log_line + "\n")
            out_file.flush()

    report = evaluator.format_report(dset)
    print(report)
    out_file.write(report + "\n")
    out_file.flush()

    summary = evaluator.summary()
    del val_dataset, val_loader, model, evaluator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def main():
    parser = argparse.ArgumentParser(description="Step 0: Measure Headroom per Dataset")
    parser.add_argument('--data', type=str, default='./datasets/', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/K/DTD/Caltech101/eurosat/Food101/UCF101/Aircraft',
                        help='test dataset(s), multiple datasets split by slash')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='maximum number of samples per dataset (default: all)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT-B/16', help='CLIP visual backbone (RN50, ViT-B/16)')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N', help='total views (1 ZS + B-1 aug)')
    parser.add_argument('-p', '--print-freq', default=200, type=int, metavar='N', help='print frequency')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default='a_photo_of_a', type=str, help='init tunable prompts')
    parser.add_argument('--load', default=None, type=str, help='path to pre-trained coop weights')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--scale', default=0.4, type=float, help='scale for adaptive alpha')
    parser.add_argument('--alpha', default=None, type=float, help='manual alpha override')
    parser.add_argument('--out', type=str, default='./results/headroom_step0.txt', help='path to log file')

    args = parser.parse_args()
    cudnn.benchmark = True

    normalize = transforms.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    base_out, _ = os.path.splitext(args.out)
    csv_path = base_out + ".csv"
    md_path = base_out + ".md"
    json_path = base_out + ".json"

    datasets_list = [d.strip() for d in args.test_sets.split('/') if d.strip()]
    results_by_dset = {}

    with open(args.out, 'w') as out_file:
        for dset in datasets_list:
            summary = evaluate_dataset(dset, args, normalize, out_file)
            results_by_dset[dset] = summary

        md_table = format_markdown_summary_table(results_by_dset, args.arch)
        print("\n" + md_table + "\n")
        out_file.write("\n" + md_table + "\n")
        out_file.flush()

    with open(md_path, 'w') as f_md:
        f_md.write(md_table + "\n")

    with open(json_path, 'w') as f_json:
        json.dump(results_by_dset, f_json, indent=2)

    if results_by_dset:
        first_summary = next(iter(results_by_dset.values()))
        fieldnames = ["dataset"] + list(first_summary.keys())
        with open(csv_path, 'w', newline='') as f_csv:
            writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
            writer.writeheader()
            for dset, s in results_by_dset.items():
                row = {"dataset": dset}
                row.update(s)
                writer.writerow(row)

    print(f"Saved summary files:\n  - Log : {args.out}\n  - MD  : {md_path}\n  - CSV : {csv_path}\n  - JSON: {json_path}")


if __name__ == '__main__':
    main()

