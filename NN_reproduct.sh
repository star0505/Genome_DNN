#!/bin/bash
echo Neural Network
python config.py

for i in {1..15}
do
	echo $i
	python NN_reproduct.py
done
