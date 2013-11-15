# Things to try
In order of subjective importance (i.e. expected score improvement / implementation time.)

1. BD: NLP - extremely naive implementation right now, how can it be improved?
1. MV: Train a random forest regressor on the data. See how it compares to GBM. If it performs well, try submitting a blend of this with GBM. 
1. BD: See whether special considerations for Chicago improve the score.
1. MV: Look for duplicates in test data. How common are they? The city often marks duplicates as closed. Doing this removes the option to vote. Can we use this?
1. BD,MV: _Why do the mean corrections in time not improve the score?_
1. BD,MV: _Why do we have large variation in correction factors between v,v and c?_
1. MV: Separate hyperparameter scans for each of {c,v,v}. More generally, think about using a different model for each of {c,v,v}.
1. BD: New feature: Submission time. Scalar feature. Should capture some of the shift in the distribution over time.
1. MV: Separate training for remote\_api\_created, perhaps even separate training sets
1. MV: Data preformatting: this has to be critical, what can we do?
1. MV: With GBM, going from 40 to 30 estimators caused a very slight improvement, would 30 to 20 cause a more significant improvement? Where on the estimators/score curve is the right place? (BD: Have you tried this, MV?)
1. BD: Think in terms of excess votes (>1).
1. Move code from notebooks when it is useful to keep.
1. Plot proportion of data in each small niche for the traning and test data sets.
1. Can we combine the mean information we have for the remoteapi with that for the first and second halves of the test data?
1. Use locality data: http://sedac.ciesin.columbia.edu/data/set/usgrid-summary-file3-2000-msa

## Useful Background
1. Read http://eaves.ca/2013/09/11/announcing-the-311-data-challenge-soon-to-be-launched-on-kaggle/ and in particular the link to the 311 standard api: http://open311.org/ . This should really help our intuition of what's going on.
1. Visualization of tag types in different cities: https://www.kaggle.com/c/the-seeclickfix-311-challenge/visualization/1299


## Observations

### 11 Nov submission (Score 0.30339, pos #8)

- With gradient boosters increasing the number of estimators can be counter-productive
- CV can be misleading




This is the repository for our entry for the See Click Predict Fix Kaggle contest.

