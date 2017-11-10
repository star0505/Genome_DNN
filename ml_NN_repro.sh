#!/bin/bash
echo Neural Network
python config.py

for i in {1..15}
do
	echo $i
	python ml_NN_repro.py
done
