# Out-of-distribution Tests Reveal Compositionality in Chess Transformers

Chess is a canonical example of a task that requires rigorous reasoning and long-term planning. Modern decision Transformers - trained similarly to LLMs - are able to learn competent gameplay, but it is unclear to what extent they truly capture the rules of chess.
To investigate this, we train a 270M parameter chess Transformer and test it on out-of-distribution scenarios, designed to reveal failures of systematic generalization.
Our analysis shows that Transformers exhibit compositional generalization, as evidenced by strong rule extrapolation: they adhere to fundamental ‘syntactic’ rules of the game by consistently choosing valid moves even in situations very different from the training data. Moreover, they also generate high-quality moves for OOD puzzles.
In a more challenging test, we evaluate the models on variants including Chess960 (Fischer Random Chess) - a variant of chess where starting positions of pieces are randomized. We found that while the model exhibits basic strategy adaptation, they are inferior to symbolic AI algorithms that perform explicit search, but gap is smaller when playing against users on Lichess. Moreover, the training dynamics revealed that the model initially learns to move only its own pieces, suggesting an emergent compositional understanding of the game.  

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

5. Download the parameters of the model:
 
Checkpoints are saved to searchless_chess/checkpoints/local

```bash
cd searchless_chess/checkpoints
./download.sh
cd ../..
```


### For training

6. The filtered dataset

downloaded directly:
```bash
./download.sh
```
OR obtained by filtering:

download the original dataset:
```bash
cd searchless_chess/data
./download.sh
cd ../..
```
then filter:
```bash
python filter_data.py 
```
7. Train the model

python searchless_chess/src/train.py

### Installing Stockfish and Fairy-Stockfish

Download and compile the latest version of Stockfish and Fairy-Stockfish (for Unix-like systems):

8. Stockfish:

```bash
git clone https://github.com/official-stockfish/Stockfish.git
cd Stockfish/src
make -j profile-build ARCH=x86-64-avx2
cd ../..
```

9. Fairy-Stockfish:

```bash
git clone https://github.com/fairy-stockfish/Fairy-Stockfish.git
cd Fairy-Stockfish/src
make -j profile-build ARCH=x86-64-avx2
cd ../..
```
### Running a tournament

```bash
python searchless_chess/src/tournament.py --num_games=100 --variant=standard
```

Adjust num_games and variant (options: standard, chess960, horde) if needed.

### Running evaluations on the OOD datasets






