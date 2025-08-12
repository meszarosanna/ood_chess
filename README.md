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

5. Filter the dataset

python filter_data.py       %for train and test folders

6. Train the model
Adjust batch_size in train.py if needed

python searchless_chess/src/train.py

7. Checkpoints

Checkpoints are saved to searchless_chess/checkpoints/local