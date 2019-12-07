#!/bin/bash
read -p "Do you want to install libraries for gpu? y or n: " ans
echo ". /home/rafid/anaconda3/etc/profile.d/conda.sh" >> ~/.bashrc
source ~/anaconda3/etc/profile.d/conda.sh
conda create -n caption_generator python=3.6
conda activate caption_generator
conda install -c anaconda scikit-learn
conda install -c conda-forge matplotlib
conda install -c anaconda pillow
conda install -c anaconda nltk
conda install -c anaconda pydot
if [ $ans = "y" ]
then
	conda install -c anaconda tensorflow-gpu
	conda install -c anaconda keras-gpu
else
	conda install -c anaconda tensorflow
	conda install -c anaconda keras
fi
