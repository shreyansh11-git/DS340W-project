# Replication of Khan et al. (2023), Tables 8 and 9

TSLA test accuracy (%), test period 2020-11-16 to 2021-12-31 (paper Table 5).
paper = reported in Tables 8 and 9; published = authors' pipeline including next-day 'Previous' features; paper_text = same, with previous-day features as the paper states.

| model               |   1-day paper |   1-day published |   1-day paper_text |   15-min paper |   15-min published |   15-min paper_text |
|:--------------------|--------------:|------------------:|-------------------:|---------------:|-------------------:|--------------------:|
| Logistic Regression |         85.51 |             77.03 |              44.88 |          90.6  |              56.54 |               49.12 |
| XGBoost             |         84.8  |             77.03 |              46.64 |          90.93 |              51.24 |               48.76 |
| Random Forest       |         84.45 |             81.63 |              45.94 |          91.27 |              55.12 |               47.7  |
| ANN                 |         84.45 |             70.67 |              53.71 |          89.93 |              56.18 |               51.24 |
| AdaBoost            |         83.74 |             83.39 |              48.76 |          90.93 |              53.36 |               46.64 |
| Decision Tree       |         83.01 |             67.84 |              48.76 |          88.1  |              49.82 |               51.94 |
| SVM                 |         82.68 |             75.62 |              47.7  |          88.59 |              55.83 |               50.88 |
| KNN                 |         79.15 |             67.84 |              47.7  |          80.53 |              53    |               49.12 |
| Naive Bayes         |         73.49 |             78.8  |              43.11 |          81.54 |              54.77 |               50.18 |
| Average (9 models)  |         82.36 |             75.54 |              47.47 |          88.05 |              53.99 |               49.51 |
| Always-up baseline  |        nan    |             55.83 |              55.83 |         nan    |              53    |               53    |
