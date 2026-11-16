# Novel techniques: comparison table

TSLA, test period 2020-11-16 to 2021-12-31 (283 trading days). Trading simulation: USD 10,000, 0.1% commission per trade, long or cash.
Buy and hold over the same period: Sharpe 1.79, return 158.7%, max drawdown -36.2%.

| Setup   | Technique                                | Models   |   Avg accuracy % |   Best accuracy % |   Days long % | Best model    | Beat always-up   |   Avg Sharpe (net) |   Best Sharpe (net) | Beat buy and hold   |
|:--------|:-----------------------------------------|:---------|-----------------:|------------------:|--------------:|:--------------|:-----------------|-------------------:|--------------------:|:--------------------|
| S0      | Paper as published (lookahead)           | 9        |            75.54 |             83.39 |         55.71 | AdaBoost      | 9 of 9           |               6.41 |                7.45 | 9 of 9              |
| S1      | Paper as written (honest)                | 9        |            47.47 |             53.71 |         55.75 | ANN           | 0 of 9           |               0.49 |                1.72 | 0 of 9              |
| S2      | + leakage fixes                          | 9        |            52.77 |             59.01 |         65.45 | SVM           | 2 of 9           |               1.15 |                2.25 | 2 of 9              |
| S3      | + scale-free features                    | 9        |            47.15 |             50.18 |         50.49 | Naive Bayes   | 0 of 9           |               0.47 |                1.24 | 0 of 9              |
| S4      | + opening return (Li)                    | 9        |            48.68 |             53    |         51.08 | Decision Tree | 0 of 9           |               0.86 |                1.82 | 1 of 9              |
| S5      | + rolling retraining (Kelly)             | 9        |            52.45 |             56.18 |         71.65 | SVM           | 1 of 9           |               1.22 |                1.8  | 1 of 9              |
| S5b     | Rolling retraining only (on S2)          | 9        |            51.67 |             58.3  |         57.91 | ANN           | 2 of 9           |               0.98 |                1.85 | 2 of 9              |
| S6      | Ridge on random features (Kelly)         | 1        |            53.71 |             53.71 |         93.64 | RFF Ridge     | 0 of 1           |               1.53 |                1.53 | 0 of 1              |
| S7a     | Opening-interval rule (Li), before costs | 1        |            48.24 |             48.24 |         52.46 | Rule          | 0 of 1           |               0.47 |                0.47 | 0 of 1              |
| S7b     | Opening-interval rule (Li), after costs  | 1        |            48.24 |             48.24 |         52.46 | Rule          | 0 of 1           |              -0.39 |               -0.39 | 0 of 1              |
|         | Always-up baseline / buy and hold        |          |            55.83 |            nan    |        100    |               |                  |               1.79 |              nan    |                     |

## Test accuracy by model (%)

| model               |     S0 |     S1 |     S2 |     S3 |     S4 |     S5 |    S5b |     S6 |    S7a |    S7b |
|:--------------------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
| Logistic Regression |  77.03 |  44.88 |  49.47 |  44.88 |  44.52 |  54.77 |  54.06 | nan    | nan    | nan    |
| Random Forest       |  81.63 |  45.94 |  49.12 |  44.52 |  49.12 |  47.7  |  49.12 | nan    | nan    | nan    |
| AdaBoost            |  83.39 |  48.76 |  53.36 |  46.64 |  49.12 |  50.18 |  49.12 | nan    | nan    | nan    |
| ANN                 |  70.67 |  53.71 |  55.83 |  48.41 |  47.35 |  55.12 |  58.3  | nan    | nan    | nan    |
| Decision Tree       |  67.84 |  48.76 |  50.18 |  45.58 |  53    |  51.24 |  43.82 | nan    | nan    | nan    |
| SVM                 |  75.62 |  47.7  |  59.01 |  48.06 |  47.7  |  56.18 |  57.6  | nan    | nan    | nan    |
| KNN                 |  67.84 |  47.7  |  51.94 |  45.94 |  45.58 |  52.65 |  52.65 | nan    | nan    | nan    |
| Naive Bayes         |  78.8  |  43.11 |  56.18 |  50.18 |  50.53 |  55.48 |  51.94 | nan    | nan    | nan    |
| XGBoost             |  77.03 |  46.64 |  49.82 |  50.18 |  51.24 |  48.76 |  48.41 | nan    | nan    | nan    |
| RFF Ridge           | nan    | nan    | nan    | nan    | nan    | nan    | nan    |  53.71 | nan    | nan    |
| Rule                | nan    | nan    | nan    | nan    | nan    | nan    | nan    | nan    |  48.24 |  48.24 |

## Trading results by model

| setup   | model               |   accuracy |   sharpe |   total_return_pct |   max_drawdown_pct |   trades |
|:--------|:--------------------|-----------:|---------:|-------------------:|-------------------:|---------:|
| S0      | Logistic Regression |      77.03 |     6.72 |            1754.09 |              -5.13 |      146 |
| S0      | Random Forest       |      81.63 |     7.08 |            2172.48 |              -6.36 |      142 |
| S0      | AdaBoost            |      83.39 |     7.45 |            2409.6  |              -5.96 |      148 |
| S0      | ANN                 |      70.67 |     5.41 |            1191.37 |             -11.12 |      104 |
| S0      | Decision Tree       |      67.84 |     5.03 |             915.22 |             -11.38 |      146 |
| S0      | SVM                 |      75.62 |     6.66 |            1701.5  |              -5.54 |      138 |
| S0      | KNN                 |      67.84 |     5.73 |            1137.64 |             -10.74 |      111 |
| S0      | Naive Bayes         |      78.8  |     6.86 |            1861.39 |              -5.54 |      148 |
| S0      | XGBoost             |      77.03 |     6.73 |            1853.7  |              -6.03 |      142 |
| S1      | Logistic Regression |      44.88 |     0.72 |              29.71 |             -36.25 |       43 |
| S1      | Random Forest       |      45.94 |    -0.33 |             -24.27 |             -43.35 |       98 |
| S1      | AdaBoost            |      48.76 |     0.46 |              12.81 |             -32.34 |       99 |
| S1      | ANN                 |      53.71 |     1.72 |             142.78 |             -36.25 |       27 |
| S1      | Decision Tree       |      48.76 |     0.46 |              12.71 |             -33.51 |      105 |
| S1      | SVM                 |      47.7  |     1.07 |              59.9  |             -40.57 |       25 |
| S1      | KNN                 |      47.7  |     0.79 |              31.02 |             -36.75 |       83 |
| S1      | Naive Bayes         |      43.11 |    -0.05 |             -11.94 |             -31.16 |       63 |
| S1      | XGBoost             |      46.64 |    -0.44 |             -24    |             -47.89 |       99 |
| S2      | Logistic Regression |      49.47 |     0.95 |              50.63 |             -36.25 |       41 |
| S2      | Random Forest       |      49.12 |     0.25 |               2    |             -39.03 |       71 |
| S2      | AdaBoost            |      53.36 |     0.84 |              33.97 |             -33.22 |      108 |
| S2      | ANN                 |      55.83 |     1.49 |              96.58 |             -44.22 |       27 |
| S2      | Decision Tree       |      50.18 |     0.97 |              45.47 |             -36.23 |       77 |
| S2      | SVM                 |      59.01 |     2.25 |             179.95 |             -20.12 |       47 |
| S2      | KNN                 |      51.94 |     0.97 |              39.7  |             -26.98 |       56 |
| S2      | Naive Bayes         |      56.18 |     1.84 |             156.62 |             -38.12 |       23 |
| S2      | XGBoost             |      49.82 |     0.75 |              29.37 |             -32.58 |      101 |
| S3      | Logistic Regression |      44.88 |     0.31 |               4.24 |             -40.47 |       69 |
| S3      | Random Forest       |      44.52 |    -0.39 |             -21.79 |             -48.61 |      115 |
| S3      | AdaBoost            |      46.64 |    -0.12 |             -12.88 |             -43.64 |      113 |
| S3      | ANN                 |      48.41 |     1.18 |              65.87 |             -39.17 |       77 |
| S3      | Decision Tree       |      45.58 |    -0.15 |             -12.1  |             -32.04 |      116 |
| S3      | SVM                 |      48.06 |     0.92 |              45.2  |             -36.25 |       30 |
| S3      | KNN                 |      45.94 |     0.06 |              -6.15 |             -40.12 |       85 |
| S3      | Naive Bayes         |      50.18 |     1.2  |              62.98 |             -27.55 |       74 |
| S3      | XGBoost             |      50.18 |     1.24 |              60.07 |             -30.4  |      112 |
| S4      | Logistic Regression |      44.52 |     0.35 |               6.35 |             -36.66 |       75 |
| S4      | Random Forest       |      49.12 |     0.97 |              42.76 |             -33.35 |      125 |
| S4      | AdaBoost            |      49.12 |    -0.19 |             -14.46 |             -37.99 |      107 |
| S4      | ANN                 |      47.35 |     0.79 |              35.72 |             -36.25 |       47 |
| S4      | Decision Tree       |      53    |     1.82 |             101.38 |             -24.68 |      129 |
| S4      | SVM                 |      47.7  |     0.85 |              39.65 |             -36.25 |       31 |
| S4      | KNN                 |      45.58 |     0.61 |              21.08 |             -27.35 |       90 |
| S4      | Naive Bayes         |      50.53 |     1.4  |              79.75 |             -21.11 |       78 |
| S4      | XGBoost             |      51.24 |     1.18 |              45.88 |             -40.91 |      129 |
| S5      | Logistic Regression |      54.77 |     1.55 |             113.3  |             -32.45 |       39 |
| S5      | Random Forest       |      47.7  |     0.14 |              -1.91 |             -53.41 |      111 |
| S5      | AdaBoost            |      50.18 |     1.09 |              47.45 |             -36.63 |      100 |
| S5      | ANN                 |      55.12 |     1.6  |             127.12 |             -36.25 |        3 |
| S5      | Decision Tree       |      51.24 |     1.36 |              76.04 |             -29.08 |      137 |
| S5      | SVM                 |      56.18 |     1.71 |             135.72 |             -36.25 |       10 |
| S5      | KNN                 |      52.65 |     1.29 |              72.03 |             -29.84 |       81 |
| S5      | Naive Bayes         |      55.48 |     1.8  |             135.29 |             -28.32 |       59 |
| S5      | XGBoost             |      48.76 |     0.45 |              11.69 |             -41.19 |      121 |
| S5b     | Logistic Regression |      54.06 |     1.56 |             110.29 |             -36.25 |       32 |
| S5b     | Random Forest       |      49.12 |     0.68 |              23.06 |             -29.78 |       93 |
| S5b     | AdaBoost            |      49.12 |     0.08 |              -4.57 |             -43.2  |       75 |
| S5b     | ANN                 |      58.3  |     1.83 |             137.54 |             -45.31 |       18 |
| S5b     | Decision Tree       |      43.82 |     0.15 |              -1.85 |             -24.32 |      114 |
| S5b     | SVM                 |      57.6  |     1.85 |             129.19 |             -37.58 |       40 |
| S5b     | KNN                 |      52.65 |     1.57 |              91.29 |             -24.61 |       52 |
| S5b     | Naive Bayes         |      51.94 |     0.57 |              17.18 |             -40.23 |       75 |
| S5b     | XGBoost             |      48.41 |     0.52 |              14.84 |             -28.61 |       97 |
| S6      | RFF Ridge           |      53.71 |     1.53 |             118.13 |             -37.16 |       33 |
| S7a     | Rule                |      48.24 |     0.47 |              11.56 |             -24.3  |      298 |
| S7b     | Rule                |      48.24 |    -0.39 |             -17.2  |             -34.02 |      298 |
