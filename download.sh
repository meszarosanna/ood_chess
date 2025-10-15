#!/bin/bash

#Download the filtered train dataset and extract it
mkdir -p searchless_chess/data/train
pip install gdown
gdown 1QVAMJC9b3hVb8aBCMyorOwHmAY3nGO1k
tar --zstd -xvf filtered_dataset.tar.zst
rm filtered_dataset.tar.zst

#Download the filtered test dataset
mkdir -p searchless_chess/data/test
cd searchless_chess/data/test
gdown 1KX6tqCrkfnBaQWxMfC3MBs5aPJAP366k
cd ../../..



