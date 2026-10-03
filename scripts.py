import subprocess

commands = ["python instance_tta.py --test_datasets eurosat --algorithm se --out ./results/se_eurosat_resnet50",
            "python instance_tta.py --test_datasets eurosat --algorithm se --out ./results/se_eurosat_vitb16",]

for i in commands:
    print(i)
    subprocess.run(i, shell=True)