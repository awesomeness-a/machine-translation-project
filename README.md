# Machine Translation with Keras 
This repository contains a machine translation program for English-Spanish translation built using deep learning with `Tensorflow`’s `Keras` API.

The dataset for this project is downloaded from [the dataset of language pairs](http://www.manythings.org/anki/).

## `preprocessing.py` 
This `Python` file contains Python code that preprocesses the text data. Here we have what is needed for `Keras` implementation:
* vocabulary sets for both input (English) and target (Spanish) data
* the total number of unique word tokens for each set
* the maximum sentence length used for each language

## `training_model.py` 
This is where we do encoder and decoder training setup, build and train the `seq2seq` model using the `Model()` function from `Keras`.

## `test_function.py`
This file contains a function that:
* accepts a `NumPy` matrix representing the test English sentence input
* uses the encoder and decoder to generate Spanish output
