#!/bin/bash
git pull
pip install -r requirements.txt
python3 train_dit.py encoded/z8r255LoVJc.h5 $@