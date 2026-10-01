# USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning

This project is cloned from [USE](https://github.com/sirujiang/USE) which is offical implementation of **"USE: A Unified Self-Ensembling Framework for Test-Time Prompt Tuning"**, accepted as a regular paper at **ICML 2026**, to reproduce results as part of my course work. 


# Experiments to RUN
- Baselines
TPT, and compare SE and USE against it.
- On ImageNet datasets, imagenet-a, imagenet-r, imagenet-sketch
- On few-shot-datasets
- eurosat, dtd, fgvc_aircraft, food101, ucf101, caltech101

- Classification accuracy:
```python instance_tta.py --test_sets <dataset> ---algorithm [se,use,tpt] -a [ViT-B/16, ResNet] --out out_file.txt```


