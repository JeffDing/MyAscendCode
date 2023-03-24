#!/bin/sh
echo [*] ViDeNN: AWGN image dataset preparation script. Run it once before training. Author: Claus Michele

cd data
rm -rf train test denoised
mkdir train train/noisy train/original test test/noisy test/original
cd ..
python add_noise_spatialCNN.py
python generate_patches_spatialCNN.py

