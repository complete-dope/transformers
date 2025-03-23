import urllib.request
import re 
import torch 

url = "https://gist.githubusercontent.com/blakesanie/dde3a2b7e698f52f389532b4b52bc254/raw/76fe1b5e9efcf0d2afdfd78b0bfaa737ad0a67d3/shakespeare.txt"


urllib.request.urlretrieve(url, "dataset/shakespeare.txt")

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import NFKC

# Initialize a BPE tokenizer
tokenizer = Tokenizer(BPE())

# Set normalizers and pre-tokenizers
tokenizer.normalizer = NFKC()
tokenizer.pre_tokenizer = Whitespace()

# Load dataset
files = ["dataset/shakespeare.txt"]  # Replace with your dataset file(s)

# Train the tokenizer
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
tokenizer.train(files, trainer)

# Save tokenizer
tokenizer.save("vocab.json")

