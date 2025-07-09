

class Tokenizer:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        """
        Encodes a string into a list of integers based on the vocabulary.
        """
        pass

    def decode(self, tokens):
        """
        Decodes a list of integers back into a string based on the vocabulary.
        """
        pass