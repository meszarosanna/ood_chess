# OOD_chess

Studying OOD generailzation of chess transformers.

## Setup instructions

1. Clone the repository: 

git clone https://github.com/meszarosanna/ood_chess.git
cd ood_chess

2. Create and activate a virtual evironment:

python -m venv chess 
source chess/bin/activate

3. Install dependencies:

pip install -r requirements.txt

4. Install jax with CUDA support

pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

5. Filter the dataset

python filter_data.py

6. Train the model

python searchless_chess/src/train.py