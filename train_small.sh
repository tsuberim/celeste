#!/bin/zsh
source .venv/bin/activate
python3 train_dit.py --max_frames 1000 --batch_size 2 --num_layers 4 --embed_dim 512 --num_heads 8 encoded/z8r255LoVJc.h5