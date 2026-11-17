# NFL Big Data Bowl 2020

Finished in roughly the top 1%, competing solo. 2,038 teams entered, and 1,598 of them were one person. See the [competition leaderboard](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/leaderboard).

Kaggle competition: how many yards will an NFL player gain after receiving a handoff?

I had to predict a probability for every possible yard total, from -99 to 99. The official score is Continuous Ranked Probability Score (CRPS). Lower is better.

This was a kernel-only competition. The code is built to run in Kaggle's nflrush environment, not as a regular local script.

## Approach

I flip the field so every play moves in the same direction, and I correct the 2017 orientation, speed, and acceleration values so they line up with later seasons.

Most of the features come from player tracking. I project where the rusher and the other players will be a short time after the handoff, then measure distances and angles between the rusher and each defender. I also keep standard play info like formation, yard line, rusher speed and acceleration, and the time between snap and handoff.

The model is a blend of two networks, trained with 5-fold GroupKFold grouped by game:

- A Keras model with two input branches, one for play-level features and one for per-defender features, with a 199-bin softmax output.
- A PyTorch model with a 199-unit sigmoid output.

The submitted prediction is 85% Keras and 15% PyTorch.

The notebook I submitted is [fast-official-nfl-solution.ipynb](fast-official-nfl-solution.ipynb). There is also a [Python export](fast-official-nfl-solution.py) of the same code. The notebook is still private on Kaggle.

## Sources

- [Wide Open Spaces](http://www.lukebornn.com/papers/fernandez_ssac_2018.pdf), the pitch control paper by Fernandez and Bornn
