# Things to try
In order of subjective importance (i.e. expected score improvement / implementation time.)

1. NLP - extremely naive implementation right now, how can it be improved?
1. Train a random forest regressor on the data. See how it compares to GBM. If it performs well, try submitting a blend of this with GBM. 
1. _Why do the mean corrections in time not improve the score?_
1. Look for duplicates in test data. How common are they? The city often marks duplicates as closed. Doing this removes the option to vote. Can we use this?
1. Separate training for remote\_api\_created, perhaps even separate training sets
1. Data preformatting: this has to be critical, what can we do?
1. With GBM, going from 40 to 30 estimators caused a very slight improvement, would 30 to 20 cause a more significant improvement? Where on the estimators/score curve is the right place? (BD: Have you tried this, MV?)
1. Move code from notebooks when it is useful to keep.
1. Plot proportion of data in each small niche for the traning and test data sets.
1. Can we combine the mean information we have for the remoteapi with that for the first and second halves of the test data?
1. Think in terms of excess votes (>1).
1. Use locality data: http://sedac.ciesin.columbia.edu/data/set/usgrid-summary-file3-2000-msa

## Observations

### 11 Nov submission (Score 0.30339, pos #8)

- With gradient boosters increasing the number of estimators can be counter-productive
- CV can be misleading




This is the repository for our entry for the See Click Predict Fix Kaggle contest.

