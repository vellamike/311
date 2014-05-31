This is the repository Mike Vella and Ben Derrett used for coding the solution that acheived [position #13](https://www.kaggle.com/c/see-click-predict-fix/leaderboard) in the Kaggle competition [See Click Fix Predict](https://www.kaggle.com/c/see-click-predict-fix/).

#WARNING

Most of the code here has highly-experimental prototype status, I would not advise anyone to use this codebase for educational purposes.

# Things to try
In order of subjective importance (i.e. expected score improvement / implementation time.)

1. BD, MV: Can we incorporate text with GBM? Perhaps we do this by reducing the number of features to the GBM.
1. MV: Tune the max_depth feature of the GBM, as suggested on Sklearn GBM page.
1. Do tfidf with max_features parameter set. e.g. = 100. Greatly speed up computation and reduce overfitting.
1. MV: Try subtting a blend of regressors. 
1. BD: See whether special considerations for Chicago improve the score.
1. MV: Look for duplicates in test data. How common are they? The city often marks duplicates as closed. Doing this removes the option to vote. Can we use this?
1. BD,MV: _Why do the mean corrections in time not improve the score?_
1. BD,MV: _Why do we have large variation in correction factors between v,v and c?_
1. BD: Speed up computating by not computing features which are never used.
1. MV: Separate hyperparameter scans for each of {c,v,v}. 
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

