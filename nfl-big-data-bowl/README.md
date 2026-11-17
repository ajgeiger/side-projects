# NFL Big Data Bowl 2020

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

The notebook I submitted is [fast-official-nfl-solution.ipynb](fast-official-nfl-solution.ipynb). There is also a [Python export](fast-official-nfl-solution.py) of the same code.

## Result

I entered on my own and finished in the top 1.2% overall (24th of 2,038 teams) with a score of 0.012568. Among solo entries I was in the top 1% (14th of 1,598).

[Competition leaderboard](https://www.kaggle.com/competitions/nfl-big-data-bowl-2020/leaderboard)

The kernel for this final submission is still private on Kaggle, so there is no public notebook link.

## Sources

- [NFL 001 feature selection](https://www.kaggle.com/coolcoder22/nfl-001-feature-selection), a notebook I used as a starting point
- [Wide Open Spaces](http://www.lukebornn.com/papers/fernandez_ssac_2018.pdf), the pitch control paper by Fernandez and Bornn
