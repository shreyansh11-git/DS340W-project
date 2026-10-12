# Khan et al. features: lagged (paper) vs lookahead (published notebook)

TSLA daily, test split 2020-03-17 to 2021-05-26. Test accuracy.

| model                   |   lagged |   same_day |   lookahead |   inflation_pts |
|:------------------------|---------:|-----------:|------------:|----------------:|
| ANN                     |    44.52 |      44.52 |       54.82 |           10.3  |
| AdaBoost                |    44.52 |      44.52 |       77.74 |           33.22 |
| Decision Tree           |    44.52 |      43.52 |       42.86 |           -1.66 |
| KNN                     |    46.18 |      43.52 |       52.82 |            6.64 |
| Logistic Regression     |    44.19 |      44.52 |       74.09 |           29.9  |
| Majority-class baseline |    55.48 |      55.48 |       55.48 |            0    |
| Naive Bayes             |    44.52 |      44.19 |       65.78 |           21.26 |
| Random Forest           |    46.84 |      44.85 |       74.75 |           27.91 |
| SVM                     |    44.52 |      44.52 |       52.16 |            7.64 |
| XGBoost                 |    47.18 |      41.86 |       68.44 |           21.26 |

## Reading the result

- With the features built as the paper specifies (lagged), no model beats the majority-class
  baseline (always predict "up", 55.48%). Most models predict "down" far too often.
- With the published notebook's forward shift (lookahead), the same models gain up to
  33 percentage points (AdaBoost 44.5% -> 77.7%, Logistic Regression 44.2% -> 74.1%,
  Random Forest 46.8% -> 74.8%). That gain comes from seeing tomorrow's bar, not from skill.
- Known issue for the next step: the scale-dependent features (SMA, volume, the three price
  differences) are out of range in the test period. Test SMA reaches 777.7 vs a train maximum
  of 131.7, so the train-fitted Min-Max scaler cannot represent them. Next: express these as
  percentages of price (scale-free), add the first-15-minute arm, and the financial simulation.
- The ANN is scikit-learn's MLPClassifier (paper: Keras). AdaBoost uses SAMME because
  SAMME.R has been removed from scikit-learn.
