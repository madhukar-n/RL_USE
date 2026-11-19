# USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning

This project is cloned from [USE](https://github.com/sirujiang/USE), the official implementation of **"USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning"** (Siru Jiang, Jian Liang, Ran He, Tieniu Tan), accepted as a regular paper at **ICML 2026**.
## Method overview

Test-time prompt tuning (TPT) adapts the text prompt of a frozen vision-language model (CLIP) on each unlabeled test image. It trains against a pseudo-label `q(x)` built from the model's predictions on augmented views of that image.

USE builds a better pseudo-label through **self-ensembling (SE)**:

- The original test image is the **weak view** and its AugMix augmentations are the **strong views**.
- The top-K most confident strong views (lowest entropy) are kept together with the weak view.
- The pseudo-label is a weighted combination of the weak and the selected strong views. The weak view gets priority, and the strong views are added adaptively.
- **Skip technique:** when every selected strong view agrees with the weak view, the optimization step is skipped.
- The prompt is optimized with **reverse cross-entropy** against this pseudo-label.

The methods compared below:

| Method | Self-ensembling used for |
|---|---|
| **SE** | The self-ensembled pseudo-label alone. In this repo, [instance_method/se.py](instance_method/se.py) returns it as the prediction without updating the prompt |
| **USE** | Adaptation **and** inference: the same ensemble produces the final prediction, so optimization and inference stay consistent |

## Results

All numbers are top-1 accuracy (%), with 1 TTA step, 64 views (batch size 64) and seed 0. They are the numbers reported in the course presentation (slides 6–10). The one exception is marked in the ViT-B/16 ImageNet table. Raw logs are in [results/](results/).

### ImageNet and its variants, ResNet-50

| Method | ImageNet | -V2 | -A | -Sketch | -R | Average |
|---|---|---|---|---|---|---|
| SE  | 64.670 | 54.220 | **30.547** | 35.536 | 58.594 | 48.713 |
| USE | **64.980** | **55.140** | 30.520 | **35.853** | **59.337** | **49.166** |

### Fine-grained datasets, ResNet-50

| Method | DTD | Caltech101 | FGVC Aircraft | EuroSAT | Oxford Flowers | UCF101 | Average |
|---|---|---|---|---|---|---|---|
| SE  | 40.662 | 87.302 | **17.642** | 23.481 | 61.389 | 60.206 | 48.447 |
| USE | **41.962** | **94.564** | 17.612 | **25.580** | **62.931** | **60.613** | **50.543** |

Average is over all six datasets.

### ImageNet and its variants, ViT-B/16

| Method | ImageNet | -V2 | -A | -Sketch | -R | Average |
|---|---|---|---|---|---|---|
| SE  | 69.288 | 64.350 | 59.760 | 48.677‡ | 77.472 | 63.909 |
| USE | **69.543** | **64.490** | **59.893** | **48.912** | **77.972** | **64.162** |

‡ The slide lists 35.536 here, which is the ResNet-50 value copied by mistake. 48.677 is from the full ViT-B/16 run ([results/se_vit_sketch.txt](results/se_vit_sketch.txt), 50,889 images).

### Fine-grained datasets, ViT-B/16

| Method | DTD | Caltech101 | FGVC Aircraft | EuroSAT | Oxford Flowers | UCF101 | Average |
|---|---|---|---|---|---|---|---|
| SE  | **46.513** | **99.500** | **25.203** | **43.914** | 67.113 | 67.128 | **58.229** |
| USE | 41.962 | 94.483 | 24.332 | 43.259 | **68.575** | **67.915** | 56.754 |

### Accuracy vs. number of TTA steps (USE)

| TTA steps | 1 | 2 | 4 | 6 |
|---|---|---|---|---|
| DTD (full test set) | 41.962 | **42.435** | 42.258 | 42.139 |
| ImageNet (1000 samples) | **77.7** | 77.4 | 77.1 | 77.1 |

### Summary

- **USE beats SE on ImageNet and its variants.** With ResNet-50 it improves the average by +0.45 points (48.71 → 49.17), and with ViT-B/16 it improves every variant (average 63.91 → 64.16).
- **With ResNet-50, USE also wins on fine-grained data.** The six-dataset average rises by +2.10 points (48.45 → 50.54), with gains on every dataset except FGVC Aircraft.
- **With ViT-B/16, the fine-grained results favour SE.** USE wins on Oxford Flowers and UCF101 but is lower on DTD, Caltech101, FGVC Aircraft and EuroSAT (average 58.23 → 56.75).
- **ImageNet-A is roughly a tie** on both backbones (differences under 0.15 points), so self-ensembling at inference brings little benefit under strong adversarial shift.
- **More TTA steps do not help.** On ImageNet, accuracy drops steadily from 77.7 (1 step) to 77.1 (4 to 6 steps). On DTD it peaks at 2 steps and then declines. Repeated optimization on a single image overfits the noisy pseudo-label, which supports the paper's default of a single step.

## Setup

### Requirements

- Python 3.12 (see [.python-version](.python-version))
- An NVIDIA GPU with CUDA 12.6 (the code calls `torch.cuda` and `torch.amp.autocast('cuda')`)
- Python packages, pinned in [requirements.txt](requirements.txt) and [Pipfile](Pipfile):
  `torch==2.13.0+cu126`, `torchvision==0.28.0+cu126`, `numpy`, `scipy`, `Pillow`, `ftfy`, `regex`, `tqdm`
- `matplotlib` is also needed for the plotting scripts in [utils/](utils/)

Install with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install matplotlib   # only for plotting
```

or with pipenv:

```bash
pipenv install && pipenv shell
```

The CLIP weights (`RN50`, `ViT-B/16`) are downloaded automatically by [clip/clip.py](clip/clip.py) on first use.

### Datasets

Datasets are read from `./datasets/` by default (change this with `--data`). The expected layout is defined in [data/datautils.py](data/datautils.py) and [data/fewshot_datasets.py](data/fewshot_datasets.py):

```
datasets/
├── imagenet/images/val/                                    # --test_sets I
├── imagenetv2/imagenetv2-matched-frequency-format-val/     # --test_sets V
├── imagenet-adversarial/imagenet-a/                        # --test_sets A
├── imagenet-sketch/images/                                 # --test_sets K
├── imagenet-rendition/imagenet-r/                          # --test_sets R
└── few-shot-datasets/
    ├── dtd/                 (images/, split_zhou_DescribableTextures.json)
    ├── caltech-101/         (101_ObjectCategories/, split_zhou_Caltech101.json)
    ├── fgvc_aircraft/
    ├── eurosat/             (2750/, split_zhou_EuroSAT.json)
    ├── oxford_flowers/
    ├── food-101/            (images/, split_zhou_Food101.json)
    └── UCF101/              (UCF-101-midframes/, split_zhou_UCF101.json)
```

The few-shot splits (`split_zhou_*.json`) follow the CoOp convention. [script.sh](script.sh) extracts the raw ImageNet ILSVRC2012 tarballs into class sub-folders.

## Running the evaluations

The entry point is [instance_tta.py](instance_tta.py):

```bash
python instance_tta.py --test_sets <SET> --algorithm <ALGO> -a <ARCH> --out <log_file>
```

| Argument | Values | Default |
|---|---|---|
| `--test_sets` | **one** dataset per run: `I`, `A`, `V`, `R`, `K`, `DTD`, `Caltech101`, `Aircraft`, `eurosat`, `Flower102`, `Food101`, `UCF101` | (must be set) |
| `--algorithm` | `use`, `se`, `tpt`, `clip` (zero-shot), `headroom` | `tpt` |
| `-a`, `--arch` | `RN50`, `ViT-B/16` | `RN50` |
| `--tta_steps` | number of optimization steps per image | `1` |
| `-b`, `--batch-size` | number of views (1 weak + 63 augmented) | `64` |
| `--selection_p` | fraction of confident views kept | `0.1` |
| `--lr` | prompt learning rate | `5e-3` |
| `--max-samples` | evaluate only the first N test images | all |
| `--data` | dataset root | `./datasets/` |
| `--out` | file that receives the running accuracy log | `output.txt` |
| `--seed`, `--gpu` | random seed, GPU id | `0`, `0` |
| `--use_feedback` | enable binary feedback for low-confidence samples (`use` only, see below) | off |
| `--feedback_ent_threshold` | normalized-entropy threshold above which feedback is queried | `0.5` |

### Commands used for the tables above

```bash
# SE and USE on every dataset and both backbones
for arch in RN50 ViT-B/16; do
  tag=${arch//\//}
  for set in I V A K R DTD Caltech101 Aircraft eurosat Flower102 UCF101; do
    for algo in se use; do
      python instance_tta.py --test_sets $set --algorithm $algo -a $arch --out results/${algo}_${set}_${tag}.txt
    done
  done
done

# Baselines
python instance_tta.py --test_sets DTD --algorithm tpt  -a RN50 --out results/tpt_dtd_resnet.txt
python instance_tta.py --test_sets DTD --algorithm clip -a RN50 --out results/clip_dtd_resnet.txt

# Accuracy vs. TTA steps
for s in 1 2 4 6; do
  python instance_tta.py --test_sets DTD --algorithm use -a RN50 --tta_steps $s --out results/use_dtd_steps$s.txt
  python instance_tta.py --test_sets I   --algorithm use -a RN50 --tta_steps $s --max-samples 1000 --out results/use_imagenet_steps$s.txt
done
```





## Extension: USE + binary feedback

### Motivation

TPT-style methods learn from the model's own predictions, through entropy minimization or a self-generated pseudo-label. Under strong domain shift the model can become *confidently wrong*, and adapting on those wrong pseudo-labels only reinforces the errors (confirmation bias). On ImageNet, many incorrect pseudo-labels have low ensemble entropy, so entropy alone cannot filter them out.

### Method

Following [Lee et al., 2025](https://arxiv.org/abs/2505.18514), the USE update is made conditional on confidence. The code is in [instance_method/use.py](instance_method/use.py).

- **High-confidence samples** (ensemble entropy ≤ threshold): the standard USE update with reverse cross-entropy against the self-ensembled pseudo-label.
- **Low-confidence samples** (ensemble entropy > threshold): the model asks for **one binary feedback signal** per sample, "is my pseudo-label correct?". It then updates the prompt with a REINFORCE loss `-r · log p(ŷ | x_weak)` with reward `r = +1` (correct) or `-1` (wrong). A correct prediction is reinforced, and a wrong one is pushed away.
- The entropy is normalized to [0, 1], and the threshold is `--feedback_ent_threshold` (0.5 in all experiments).
- In this offline evaluation the feedback is simulated from the dataset label. The run log ends with how many samples asked for feedback.

```bash
python instance_tta.py --test_sets I       --algorithm use -a RN50 --max-samples 5000 --use_feedback --feedback_ent_threshold 0.5 --out results/use_feedback_imagenet_rn50.txt
python instance_tta.py --test_sets DTD     --algorithm use -a RN50 --use_feedback --feedback_ent_threshold 0.5 --out results/use_feedback_dtd_rn50.txt
python instance_tta.py --test_sets eurosat --algorithm use -a RN50 --use_feedback --feedback_ent_threshold 0.5 --out results/use_feedback_eurosat_rn50.txt
```

### Results (ResNet-50, top-1 accuracy %)

| Dataset | USE | USE + binary feedback | Gain |
|---|---|---|---|
| ImageNet (first 5000 samples) | 67.10 | **74.84** | +7.74 |
| DTD | 41.96 | **49.17** | +7.21 |
| EuroSAT | 25.58 | **28.96** | +3.38 |

Running accuracy as more test samples are seen:

| DTD, samples seen | 200 | 400 | 800 | 1200 | 1600 | 1692 |
|---|---|---|---|---|---|---|
| USE | 32.50 | 45.75 | 36.62 | 40.42 | 42.06 | 41.96 |
| USE + feedback | 36.00 | 50.50 | 42.50 | 47.08 | 48.88 | 49.17 |

| ImageNet, samples seen | 1000 | 2000 | 3000 | 4000 | 5000 |
|---|---|---|---|---|---|
| USE | 77.70 | 72.85 | 66.40 | 64.67 | 67.10 |
| USE + feedback | 83.90 | 79.90 | 74.70 | 72.67 | 74.84 |

- Feedback improves accuracy on all three datasets, by up to about 7.7 points.
- The gap shows up early (within the first 200–400 samples) and stays throughout the run, so a small number of feedback queries is enough to stop errors from reinforcing themselves.
- Feedback uses ground-truth information for the queried samples, so these numbers are not directly comparable with fully unsupervised TTA. The fraction of queried samples should be reported alongside them.

### Future work

- Every test image needs 64 augmented views, which is expensive. Reducing the number of views would lower the cost of USE (and of episodic TTA in general).
- Sweep `--feedback_ent_threshold` to trade feedback budget against accuracy.

## References

1. S. Jiang, J. Liang, R. He, and T. Tan, "USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning," ICML 2026. arXiv:2607.03900.
2. T. Lee, S. Chottananurak, J. Kim, J. Shin, T. Gong, and S.-J. Lee, "Test-time adaptation with binary feedback," arXiv:2505.18514, 2025.
