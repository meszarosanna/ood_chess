#!/bin/bash

mkdir -p searchless_chess/data/train
pip install gdown
gdown 1yqzQLRjEwZrx-thXHIzBpGy8530jZl8l
tar --zstd -xvf filtered_dataset.tar.zst