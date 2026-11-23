# TDFormer: A novel triple decoupled transformer for accurate multi-step wind power forecasting

> **Paper authors**  
> Lei Liu, Xiangyue Liu, Jie Huang, Tianqi Li, Qiuju Chen, and Bin Li.

This repo contains the code and data from our paper published in Engineering Applications of Artificial Intelligence.

[Paper](https://doi.org/10.1016/j.engappai.2026.115310) 

## 1. Abstract

Accurate wind power forecasting (WPF) is crucial for ensuring the stable operation of power systems. Long-term effective forecasting can optimize wind power generation planning and reduce operational costs. Although deep learning advancements have significantly improved wind power forecasting accuracy, existing methods often struggle to effectively separate the complex relationships between variables, features, and time dimensions in multivariate time-series data, limiting their predictive performance. To address this issue, this paper proposes a novel model, named Triple Decoupled Transformer (TDFormer), for multi-step wind power forecasting. The model incorporates a novel Triple Decoupled Attention (TDA) mechanism to independently decouple input data across three dimensions—variables, features, and time—enabling more efficient and precise feature extraction. TDFormer adopts an encoder-decoder architecture to integrate historical data from multiple variables and future time sequences, making it ideal for multi-step wind power forecasting. Extensive experiments conducted on real-world datasets from wind farms with varying capacities and geographical locations demonstrate that the proposed TDFormer significantly outperforms state-of-the-art (SOTA) methods, especially in scenarios involving longer forecasting horizons. In addition, the model demonstrated excellent generalization ability and practical application potential in the transfer learning experiments, making it particularly suitable for newly established wind farms with limited data, and offering significant application value. The source code and datasets are available at https://github.com/USTC-AI4EEE/TDFormer.

## 2. Environment Setup

We recommend Linux with Python 3.10. Create and activate a Conda environment, then install the dependencies from the repository root:

```bash
conda create -n tdformer python=3.10 pip -y
conda activate tdformer
export PYTHONNOUSERSITE=1
python -m pip install -r environment/requirements-pip.txt
```

The main dependencies are PyTorch 1.13.1, PyTorch Lightning 1.9.0, PyTorch Forecasting 0.10.3, NumPy 1.23.5, pandas 1.5.3, and scikit-learn 1.1.3.

For GPU training, use an NVIDIA GPU with a compatible driver. The original environment used torch `1.13.1+cu117`; to closely match the original CUDA environment, install this build before installing the complete dependency set:

```bash
python -m pip install 'torch==1.13.1+cu117' --extra-index-url https://download.pytorch.org/whl/cu117
python -m pip install -r environment/requirements-pip.txt
```

Please use the specified dependency versions; PyTorch Lightning 2.x is not compatible with the current training code.

## 3. Dataset

The processed wind farm dataset is provided in [`data/processed/spring.csv`](data/processed/spring.csv). It contains 2,208 hourly records from March 1 to May 31, 2022. 

Each record includes the station ID, timestamp, wind power output (MW), and ten meteorological variables: air pressure, relative humidity, cloud cover, temperature, radiation, precipitation, and wind speed and direction at heights of 10 m and 100 m.

The data are split chronologically into training, validation, and test sets with a ratio of 80% / 10% / 10%, corresponding to 1,766 / 221 / 221 records. Input and prediction windows are constructed separately within each split. The provided configuration uses 24 hours of historical input to predict the next 48 hours.

## 4. Usage

Run from the repository root. Train all four variants on GPU 0 (change `--gpu` as needed):

```bash
conda activate tdformer
export PYTHONNOUSERSITE=1

for variant in full wo_tan wo_fan wo_van; do
  python code/main_train_pytorch_forecasting_TDFormer.py \
    --gpu 0 \
    --module_ablation "$variant" \
    --metrics_json "results/horizon_48h/metrics/$variant/seed_2024.json" || break
done
```

Evaluate the saved predictions for all four variants:

```bash
python code/evaluate_48h_prediction_prefixes.py --results-dir results
```

## 5. Experimental Results

The following results are obtained on the spring dataset from the wind farm. The model uses 24 hours of historical data to predict the next 48 hours. Each table cell reports MAE / RMSE (%), normalized by installed capacity and multiplied by 100.

| Evaluation interval | Full | w/o TAN | w/o FAN | w/o VAN |
|---|---|---|---|---|
| First 1h | **8.075** / **10.504** | 9.418 / 11.925 | 8.714 / 11.573 | 9.891 / 13.798 |
| First 8h | **9.164** / **12.258** | 10.420 / 13.202 | 9.521 / 12.579 | 10.456 / 14.093 |
| First 16h | **9.483** / **12.589** | 11.087 / 13.979 | 10.113 / 13.102 | 10.846 / 14.466 |
| First 24h | **9.651** / **12.795** | 11.214 / 14.101 | 10.344 / 13.352 | 11.205 / 14.794 |
| First 48h | **9.974** / **13.405** | 10.540 / 13.530 | 10.371 / 13.683 | 11.180 / 14.685 |

## 6. Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{LIU2026115310,
title = {TDFormer: A novel triple decoupled transformer for accurate multi-step wind power forecasting},
journal = {Engineering Applications of Artificial Intelligence},
volume = {181},
pages = {115310},
year = {2026},
issn = {0952-1976},
doi = {https://doi.org/10.1016/j.engappai.2026.115310},
url = {https://www.sciencedirect.com/science/article/pii/S0952197626015940},
author = {Lei Liu and Xiangyue Liu and Jie Huang and Tianqi Li and Qiuju Chen and Bin Li}
}
```

If you have any problems, contact me via liulei13@ustc.edu.cn.
