# BERT-Style Language Model Training from Scratch (Turkish OSCAR)

This project demonstrates how to train a BERT-style language model **from scratch** using the Turkish portion of the OSCAR dataset. No pre-trained models are used — both the tokenizer and the model are built and trained entirely from the ground up.

## Project Overview

- Uses the `oscar` dataset (`unshuffled_deduplicated_tr`) via Hugging Face Datasets
- Trains a custom WordLevel tokenizer from scratch
- Builds a small BERT-style masked language model
- Uses Hugging Face Transformers and PyTorch
- Saves the trained model and tokenizer for later use

## Requirements

Install the required packages:

```bash
pip install transformers datasets tokenizers torch
