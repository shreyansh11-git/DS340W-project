# HELD-OUT VALIDATION SET - DO NOT TOUCH

152 trading days, 2021-05-27 to 2021-12-31.

This is the unseen set. It must not be read, plotted, summarised, or used for
model selection, feature choice, or hyperparameter tuning.

Open it exactly once, at the very end, when the pipeline is bug-free and the
Train/Test results are final. Every look at this data costs you the ability to
claim the final number is honest.

`src/data.py` refuses to load it unless you pass `allow_holdout=True`.
