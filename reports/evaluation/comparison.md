| Model | Test accuracy | Test macro-F1 | CPU latency | Size | Parameters | Val F1 baseline → tuned |
|---|---|---|---|---|---|---|
| Custom CNN | 0.655 ± 0.049 | 0.660 ± 0.055 | 11.8 ms | 4.7 MB | 1.17 M | 1.00 → 1.00 |
| MobileNetV3-small | 0.759 ± 0.129 | 0.743 ± 0.140 | 3.1 ms | 6.2 MB | 1.52 M | 0.65 → 0.91 |
| ResNet18 | 0.989 ± 0.016 | 0.990 ± 0.014 | 9.7 ms | 44.8 MB | 11.18 M | 1.00 → 1.00 |

Test set: 29 images, never used for training or tuning. Mean ± std over seeds [42, 43, 44].
