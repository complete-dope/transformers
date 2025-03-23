# This is to test whether the tokenizer is working or not

from tokenizers import Tokenizer

test_sentence = "Et tu, Brute?"

tokenizer = Tokenizer.from_file("vocab.json")
encoded = tokenizer.encode(test_sentence)


print(f"The sentence '{test_sentence}' have tokens : {encoded.tokens} and the token ID's are {encoded.ids}")

