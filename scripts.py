import subprocess

commands = ["python instance_tta.py --test_sets Caltech101 --algorithm se --out ./results/se_eurosat_vitb16",
            "python instance_tta.py --test_sets Caltech101 --algorithm se --out ./results/se_caltech101_resnet50"]

for i in commands:
    print(i)
    subprocess.run(i, shell=True)