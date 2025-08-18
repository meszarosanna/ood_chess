# OOD_chess

Studying OOD generailzation of chess transformers.

## Setup instructions

1. Clone the repository

git clone https://github.com/meszarosanna/ood_chess.git
cd ood_chess

2. Create and activate a virtual evironment and set

python3 -m venv chess 
source chess/bin/activate
export PYTHONPATH=$PYTHONPATH:(pwd)

3. Install dependencies:

pip install -r requirements.txt

If GPU available:

pip install --upgrade "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


5. Filtered dataset

~/.download.sh

OR

python filter_data.py

6. Train the model

python searchless_chess/src/train.py

7. Checkpoints

Checkpoints are saved to searchless_chess/checkpoints/local