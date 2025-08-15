#!/bin/bash

cd ood_chess
mkdir searchless_chess/data/train
pip install gdown
gdown 1ga9EPgP6O_Y23TJaNtpXrUItYsuAaxcb
tar --zstd -xvf filtered_dataset.tar.zst