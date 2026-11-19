# Step 0 Headroom Summary (`ViT-B/16`)

| Dataset | N | 1. SE $q$ (USE) | 1b. SE $q$ (Quad) | 2a. Oracle $q$ (SE-Mixed) | 2b. Oracle $q$ (Pure Aug) | 2c. Oracle $q$ (Pure All) | 3a. ZS View 0 | 3b. Min-Ent Single Aug | 3c. Best Fixed View | 3d. Oracle Single View (All) | Headroom (Mixed - SE) | Headroom (Pure - SE) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DTD | 1000 | 43.20% | 42.90% | 49.70% | 68.10% | 68.30% | 40.50% | 42.50% | 40.50% | 68.30% | +6.50% | +25.10% |
| Caltech101 | 1000 | 97.60% | 97.70% | 98.00% | 99.80% | 99.80% | 96.30% | 96.10% | 96.30% | 99.80% | +0.40% | +2.20% |
| eurosat | 1000 | 58.60% | 57.60% | 76.40% | 95.10% | 95.40% | 64.80% | 42.20% | 64.80% | 95.40% | +17.80% | +36.80% |
