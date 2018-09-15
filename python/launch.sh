#!/bin/bash

git checkout LeNet1
python3 python/train.py
git add .
git commit -m "Train LeNet1"

git checkout LeNet2
python3 python/train.py
git add .
git commit -m "Train LeNet2"

git checkout LeNet3
python3 python/train.py
git add .
git commit -m "Train LeNet3"

git checkout AllCNN32
python3 python/train.py
git add .
git commit -m "Train AllCNN32"