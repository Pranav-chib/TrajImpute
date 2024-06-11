# <p align=center>`Pedestrian Trajectory Prediction with Missing Data: Datasets, Imputation, and Benchmarking`<br>
<p align=center> This repo contains datasets and code files for Pedestrian Trajectory Prediction with Missing Data: Datasets, Imputation, and Benchmarking.<br>


## Table of Contents

- [TrajImpute Dataset](#TrajImpute-Dataset)
- [Trajectory Imputataion Benchmarking](#Language-Guided-Network)
- [Trajectory Prediction Benchmarking](#Language-Guided-Network)

***

  # <p align=center> `TrajImpute Dataset`<br>
  The dataset can be downloaded from [**Download Link**](https://drive.google.com/drive/folders/1AoKeBmJQPEiUAmAn4qpefRMc89XxHgRL?usp=sharing). The structure of the TrajImpute dataset follows a dictionary format with specific keys:

<p align="center">
<img src="/TrajImpute.png" width="450" height="500"/>
<p><br>

***
  # <p align=center> `Trajectory Imputataion Benchmarking` <br>

  ## Results for Various Imputation Methods on Different Datasets

Results obtained for various imputation methods on the ETH-M, HOTEL-M, UNIV-M, ZARA1-M, and ZARA2-M subsets of TrajImpute with the easy protocol ($0 \leq \text{missing} \leq 4$) and the hard protocol ($4 \leq \text{missing} \leq 7$). The reported results show that SITES performs relatively better when imputing missing values. `M' refers to missing, indicating that the subset contains missing observed coordinates. The Imputation model file can be download at [**Download Link**](https://drive.google.com/drive/folders/16A3yo-FKzuBmMbr3eRYgiQCJ4inaSf1j?usp=sharing).

| Datasets | Methods         | Metrics | [**Transformer**](https://github.com/jadore801120/attention-is-all-you-need-pytorch) | [**US-GAN**](https://github.com/zjuwuyy-DL/Generative-Semi-supervised-Learning-for-Multivariate-Time-Series-Imputation) | [**BRITS**](https://github.com/caow13/BRITS)  | [**M-RNN**](https://github.com/jsyoon0823/MRNN)  | [**TimesNet**](https://github.com/thuml/TimesNet) | [**SAITS**](https://github.com/WenjieDu/SAITS/tree/main)  |
|----------|-----------------|---------|-------------|--------|--------|--------|----------|--------|
| ETH-M    | Easy-impute     | MAE     | 3.1318      | 0.6467 | 1.4287 | 5.2558 | 1.1353   | 0.5031 |
|          |                 | MSE     | 19.4576     | 1.8055 | 4.7339 | 35.3738| 4.9441   | 0.9909 |
|          |                 | RMSE    | 4.4111      | 1.3437 | 2.1758 | 5.9476 | 2.2235   | 0.9954 |
|          |                 | MRE     | 0.5236      | 0.1081 | 0.2389 | 0.8787 | 0.1898   | 0.0841 |
|  ETH-M  | Hard-impute     | MAE     | 3.2249      | 3.0451 | 3.0371 | 5.3309 | 1.3656   | 0.9965 |
|          |                 | MSE     | 19.5948     | 18.0716| 17.9457| 35.5047| 4.9937   | 2.5934 |
|          |                 | RMSE    | 4.7926      | 4.2511 | 4.2362 | 5.9965 | 2.5054   | 1.6104 |
|          |                 | MRE     | 0.5734      | 0.5100 | 0.5087 | 0.8962 | 0.2287   | 0.1669 |
|          |                 |       |        |   |  |   |     |   |
| HOTEL-M  | Easy-impute     | MAE     | 8.8847      | 2.6327 | 3.9033 | 3.2133 | 7.4037   | 2.1930 |
|          |                 | MSE     | 91.5550     | 13.5993| 23.1058| 20.0857| 124.5438 | 8.7460 |
|          |                 | RMSE    | 9.5684      | 3.6877 | 4.8068 | 4.4817 | 11.1599  | 2.9574 |
|          |                 | MRE     | 2.9468      | 0.8732 | 1.2946 | 1.0658 | 2.4556   | 0.7274 |
| HOTEL-M   | Hard-impute     | MAE     | 8.9096      | 7.8833 | 7.6057 | 3.2443 | 7.9484   | 2.6050 |
|          |                 | MSE     | 92.2607     | 75.9804| 72.0169| 20.2543| 106.7010 | 16.0168|
|          |                 | RMSE    | 9.6478      | 8.7167 | 8.4863 | 4.5005 | 11.3296  | 4.0021 |
|          |                 | MRE     | 2.8866      | 2.6127 | 2.5207 | 1.1686 | 2.6343   | 0.8634 |
|          |                 |       |        |   |  |   |     |   |
| UNIV-M   | Easy-impute     | MAE     | 3.0410      | 0.9158 | 1.0171 | 6.8380 | 0.6713   | 0.1939 |
|          |                 | MSE     | 14.0163     | 2.6297 | 2.9769 | 56.9715| 0.7631   | 0.0697 |
|          |                 | RMSE    | 3.7438      | 1.6216 | 1.7254 | 7.5479 | 0.8736   | 0.2639 |
|          |                 | MRE     | 0.3905      | 0.1176 | 0.1306 | 0.8780 | 0.0862   | 0.0249 |
| UNIV-M    | Hard-impute     | MAE     | 3.9795      | 1.9430 | 1.8028 | 6.9148 | 0.9421   | 0.6158 |
|          |                 | MSE     | 15.4244     | 6.1815 | 5.4057 | 57.6533| 1.5827   | 0.6003 |
|          |                 | RMSE    | 3.9639      | 2.4863 | 2.3250 | 7.7268 | 1.2581   | 0.7748 |
|          |                 | MRE     | 1.0326      | 0.2495 | 0.2315 | 0.9751 | 0.1210   | 0.0791 |
|          |                 |       |        |   |  |   |     |   |
| ZARA1-M  | Easy-impute     | MAE     | 2.6288      | 0.4832 | 0.7307 | 5.1152 | 0.3125   | 0.2054 |
|          |                 | MSE     | 10.0109     | 0.8599 | 1.2306 | 34.9869| 0.1768   | 0.0775 |
|          |                 | RMSE    | 3.1640      | 0.9273 | 1.1093 | 5.9150 | 0.4204   | 0.2784 |
|          |                 | MRE     | 0.4326      | 0.0795 | 0.1202 | 0.8417 | 0.0514   | 0.0338 |
|ZARA1-M   | Hard-impute     | MAE     | 2.7532      | 2.2846 | 2.3140 | 5.1921 | 0.5699   | 0.6277 |
|          |                 | MSE     | 10.1228     | 7.8216 | 8.0351 | 35.7821| 0.6327   | 0.8287 |
|          |                 | RMSE    | 3.1816      | 2.7967 | 2.8346 | 5.9976 | 0.7955   | 0.9103 |
|          |                 | MRE     | 0.4463      | 0.3756 | 0.3805 | 0.8673 | 0.0937   | 0.1032 |
|          |                 |       |        |   |  |   |     |   |
| ZARA2-M  | Easy-impute     | MAE     | 2.1301      | 0.3861 | 0.5556 | 5.0905 | 0.2409   | 0.1314 |
|          |                 | MSE     | 7.3276      | 0.6212 | 0.8292 | 31.5674| 0.1329   | 0.0385 |
|          |                 | RMSE    | 2.7070      | 0.7882 | 0.9106 | 5.6185 | 0.3645   | 0.1963 |
|          |                 | MRE     | 0.3524      | 0.0639 | 0.0919 | 0.8422 | 0.0399   | 0.0217 |
| ZARA2-M   | Hard-impute     | MAE     | 2.2840      | 1.8605 | 1.8051 | 5.1698 | 0.5031   | 0.3632 |
|          |                 | MSE     | 7.6342      | 5.8511 | 5.5953 | 32.3531| 0.6525   | 0.4313 |
|          |                 | RMSE    | 2.8630      | 2.4189 | 2.3654 | 5.8994 | 0.8077   | 0.6567 |
|          |                 | MRE     | 0.3735      | 0.3041 | 0.2951 | 0.8465 | 0.0823   | 0.0593 |


***

# <p align=center> `Trajectory Prediction Benchmarking` <br>

  ## Results obtained for various trajectory prediction methods on the imputed subsets of TrajImpute

We report the ADE/FDE for the trajectory prediction task on the clean, soft imputed, and hard imputed protocols. `Clean' refers to a subset with no missing coordinates. Performance degradation occurs when trajectory prediction is performed on the hard imputed subsets. The Trajectory Prediction baseline model file can be download at [**Download Link**](https://drive.google.com/drive/folders/16A3yo-FKzuBmMbr3eRYgiQCJ4inaSf1j?usp=sharing).

| Datasets | Baselines       | [**GraphTern**](https://github.com/InhwanBae/GPGraph) | [**LBEBM-ET**](https://github.com/InhwanBae/EigenTrajectory)  | [**SGCN-ET**](https://github.com/InhwanBae/EigenTrajectory)  | [**EQmotion**](https://github.com/MediaBrain-SJTU/EqMotion)  | [**TUTR**](https://github.com/lssiair/TUTR)     | [**GPGraph**](https://github.com/InhwanBae/GPGraph)   |
|----------|-----------------|-----------|-----------|----------|-----------|----------|-----------|
| ETH      | Clean           | 0.42/0.58 | 0.36/0.53 | 0.36/0.57| 0.40/0.61 | 0.40/0.61| 0.43/0.63 |
|          | Easy-impute     | 0.77/0.74 | 0.37/0.55 | 0.42/0.71| 0.46/0.62 | 0.54/0.73| 0.45/0.75 |
|          | Hard-impute     | 0.78/0.77 | 0.85/1.07 | 1.07/1.44| 0.47/0.63 | 1.12/1.53| 0.92/0.93 |
|          |                 |           |           |          |           |          |            |
| Hotel    | Clean           | 0.14/0.23 | 0.12/0.19 | 0.13/0.21| 0.12/0.18 | 0.11/0.18| 0.18/0.30 |
|          | Easy-impute     | 0.15/0.25 | 0.13/0.20 | 0.14/0.23| 0.65/0.68 | 1.31/1.66| 0.19/0.31 |
|          | Hard-impute     | 1.68/1.42 | 3.31/4.13 | 3.21/3.92| 0.72/0.74 | 3.36/3.95| 1.89/1.70 |
|          |                 |           |           |          |           |          |            |
| UNV      | Clean           | 0.26/0.45 | 0.24/0.43 | 0.24/0.43| 0.23/0.43 | 0.23/0.42| 0.24/0.42 |
|          | Easy-impute     | 0.27/0.47 | 0.30/0.51 | 0.29/0.51| 0.37/0.61 | 0.31/0.49| 0.25/0.44 |
|          | Hard-impute     | 0.50/0.51 | 0.64/1.01 | 0.77/1.21| 0.39/0.70 | 0.59/0.85| 0.53/0.50 |
|          |                 |           |           |          |           |          |            |
| ZARA1    | Clean           | 0.21/0.37 | 0.19/0.33 | 0.20/0.35| 0.18/0.32 | 0.18/0.34| 0.17/0.31 |
|          | Easy-impute     | 0.22/0.38 | 0.20/0.35 | 0.22/0.38| 0.27/0.43 | 0.24/0.41| 0.18/0.32 |
|          | Hard-impute     | 0.96/1.25 | 0.37/0.60 | 0.61/0.97| 0.28/0.44 | 0.50/0.77| 0.58/0.45 |
|          |                 |           |           |          |           |          |            |
| ZARA2    | Clean           | 0.17/0.29 | 0.14/0.24 | 0.15/0.26| 0.13/0.23 | 0.13/0.25| 0.15/0.29 |
|          | Easy-impute     | 0.18/0.30 | 0.16/0.27 | 0.17/0.29| 0.36/0.54 | 0.25/0.37| 0.29/0.30 |
|          | Hard-impute     | 0.37/0.44 | 0.27/0.43 | 0.41/0.63| 0.37/0.55 | 0.33/0.50| 0.36/0.34 |

