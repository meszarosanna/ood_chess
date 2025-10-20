# Out-of-distribution Tests Reveal Compositionality in Chess Transformers

Chess is a canonical example of a task that requires rigorous reasoning and long-term planning. Modern decision Transformers - trained similarly to LLMs - are able to learn competent gameplay, but it is unclear to what extent they truly capture the rules of chess.
To investigate this, we train a 270M parameter chess Transformer and test it on out-of-distribution scenarios, designed to reveal failures of systematic generalization.
Our analysis shows that Transformers exhibit compositional generalization, as evidenced by strong rule extrapolation: they adhere to fundamental ‘syntactic’ rules of the game by consistently choosing valid moves even in situations very different from the training data. Moreover, they also generate high-quality moves for OOD puzzles.
In a more challenging test, we evaluate the models on variants including Chess960 (Fischer Random Chess) - a variant of chess where starting positions of pieces are randomized. We found that while the model exhibits basic strategy adaptation, they are inferior to symbolic AI algorithms that perform explicit search, but gap is smaller when playing against users on Lichess. Moreover, the training dynamics revealed that the model initially learns to move only its own pieces, suggesting an emergent compositional understanding of the game.  

## Contents

```
.
|
├── BayesElo                                            - Elo computation (need to be installed)
|
├── datasets                                            - Datasets for evaluation
|   ├── all_ordering_puzzles.csv
|   ├── chess960_puzzles.csv
|   ├── filtered_behavioral_cloning_test_data.bag
|   ├── id_puzzles.csv
|   ├── knights_and_rooks.csv
|   ├── more_pieces_puzzles.csv
|   ├── ood_puzzles.csv
|   └── same_color_puzzles.csv
|
├── evaluation                                          
|   ├── board_figues.py                                 - Board figure creation
|   ├── eval_ood_puzzles.py                             - Script for OOD evaluation
|   ├── filter_data.py                                  - Filtering the train\test data
|   ├── knights_puzzles.py                              - Knight&Rooks dataset creation
|   ├── lichess_bot.py                                  - Script for playing games with out model as a lichess bot
|   ├── more_pieces_and_same_color.py                   - More pieces and Same color datasets creation
|   └── training_dynamics.py                            - Script for creating the training dynamics figures
|
├── Fairy-Stockfish                                     - Fairy-Stockfish (needs to be installed)
|
├── searchless_chess
|   ├── checkpoints                     
|   |   ├── local > behavioral_cloning > 10000000       - Model checkpoint (needs to be downloaded)
|   |   └── download.sh                                 - Downloads the model checkpoint
|   |
|   ├── data                            
|   |    ├── test
|   |    ├── train
|   |    ├── download.sh                                - Downloads the unfiltered train/test data
|   |    ├── chess960_openings.csv                      - Openings for Chess960 tournament
|   |    └── eco_openings.csv                           - Openings for Standard tournament
|   |
|   ├── src
|   |   ├── engines
|   |   |   ├── constants.py                            - Engine constants
|   |   |   ├── engine.py                               - Engine interface
|   |   |   ├── fairy_stockfish_engine.py               - Fairy-Stockfish engine
|   |   |   ├── neural_engines.py                       - Neural engines
|   |   |   └── stockfish_engine.py                     - Stockfish engine
|   |   |
|   |   ├── bagz.py                                     - Readers for our .bag data files
|   |   ├── config.py                                   - Experiment configurations
|   |   ├── constants.py                                - Constants, interfaces, and types
|   |   ├── data_loader.py                              - Data loader
|   |   ├── metrics_evaluator.py                        - Metrics evaluator
|   |   ├── tokenizer.py                                - Chess board tokenization
|   |   ├── tournament.py                               - Elo tournament script
|   |   ├── train.py                                    - Training + evaluation script
|   |   ├── training.py                                 - Training loop
|   |   ├── training_utils.py                           - Training utility functions
|   |   ├── transformer.py                              - Decoder-only Transformer
|   |   └── utils.py                                    - Utility functions
|
├── Stockfish                                           - Stockfish (needs to be installed)
|
├── download.sh                                         - Downloads the filtered train/test datasets
├── README.md
└── requirements.txt                                    - Dependencies
```

## Setup instructions

### Basic setup

1. Clone the repository

```bash
git clone https://github.com/meszarosanna/ood_chess.git
```

2. Create and activate a virtual evironment

```bash
python3 -m venv chess 
source chess/bin/activate
cd ood_chess
export PYTHONPATH=$PYTHONPATH:(pwd)
```

3. Install dependencies:

This repository requires Python 3.12.

```bash
pip install -r requirements.txt
```

If GPU available:

```bash
pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4. Download the parameters of the model:
 
Checkpoints are saved to searchless_chess/checkpoints/local

```bash
cd searchless_chess/checkpoints
./download.sh
cd ../..
```

### For training

5. The filtered dataset can be downloaded directly:
```bash
./download.sh
```

6. Train the model

```bash
python searchless_chess/src/train.py
```

### Installing Stockfish and Fairy-Stockfish

Download and compile the latest version of Stockfish and Fairy-Stockfish (for Unix-like systems):

7. Stockfish:

```bash
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src
make -j profile-build ARCH=x86-64-avx2
cd ../..
```

8. Fairy-Stockfish:

```bash
git clone https://github.com/fairy-stockfish/Fairy-Stockfish.git
cd Fairy-Stockfish/src
make -j profile-build ARCH=x86-64-avx2
cd ../..
```

### Installing BayesElo

To compute the Elos for the different agents, we use [BayesElo](https://www.remi-coulom.fr/Bayesian-Elo/):

```bash
wget https://www.remi-coulom.fr/Bayesian-Elo/bayeselo.tar.bz2
tar -xvjf bayeselo.tar.bz2
cd BayesElo
make bayeselo
cd ..
```

### Running a tournament

To compute the Elo for the model, run the tournament to play games and then compute the Elo for the PGN file generated by the tournament:

```bash
python searchless_chess/src/tournament.py --num_games=100 --variant=standard
```

Adjust num_games and variant (options: standard, chess960, horde).

```bash
cd BayesElo

./bayeselo
> ...
ResultSet>readpgn filename.pgn
> N game(s) loaded, 0 game(s) with unknown result ignored.
ResultSet>elo
ResultSet-EloRating>mm
> 00:00:00,00
ResultSet-EloRating>exactdist
> 00:00:00,00
ResultSet-EloRating>ratings
> ...

cd ..
```

### Running evaluations on the OOD datasets

```bash
python evaluation/eval_ood_puzzles.py --input_file_name=ood_puzzles.csv
```
Options for input_file_names:
- id_puzzles.csv
- ood_puzzles.csv
- filtered_behavioral_cloning_test_data.bag
- more_pieces_puzzles.csv
- same_color_puzzles.csv
- all_ordering_puzzles.csv
- chess960_puzzles.csv
- knights_and_rooks.csv

### Datasets creation
- searchless_chess/data/train/behavioral_cloning_data.bag and searchless_chess/data/test/behavioral_cloning_data.bag:

    download the original dataset
```bash
    cd searchless_chess/data
    ./download.sh
    cd ../..
```

- searchless_chess/data/train/filtered_behavioral_cloning_data.bag and searchless_chess/data/test/filtered_behavioral_cloning_data.bag:

    to filter the datasets run the following for the train dataset and test dataset
```bash
    python evaluation/filter_data.py 
```

- datasets/more_pieces_puzzles.csv and datasets/same_color_puzzles.csv:
```bash
    python evaluation/more_pieces_and_same_color.py
```

- datasets/knights_and_rooks.csv:
```bash
    python evaluate/knights_puzzles.py
```

### Trainig dynamics
To recreate the training dynamics, run
```bash
python evaluate/training_dynamics.py
```

### Lichess bot
After registrating the model to lichess and upgrading it to bot account, insert the API token to the following script and run
```bash
python evaluation/lichess_bot.py
```
 
### Create board figures
The boards in the paper were created by running
```bash
python evaluate/board_figures.py
```

## License and disclaimer

Copyright 2025 Anna Meszaros and DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

The model weights are licensed under Creative Commons Attribution 4.0 (CC-BY).
You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Some portions of the dataset are in the public domain by a
Creative Commons CC0 license from lichess.org.
The remainder of the dataset is licensed under
Creative Commons Attribution 4.0 (CC-BY).
You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode.

Unless required by applicable law or agreed to in writing, software and
materials distributed under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.


