# Copyright Meszaros et al.
#
# This script downloads the pretrained model on the filtered dataset.
# For simplicity, the file is in the searchless_chess folder, but it was created by Meszaros et al.
#

mkdir -p local/behavioral_cloning
cd local/behavioral_cloning
gdown --folder https://drive.google.com/drive/folders/1X-LUMs2Zv8OpbX5tlfjgfMnFkKJW9S-r?usp=sharing
cd ../..