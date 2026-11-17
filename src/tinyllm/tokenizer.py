
import sentencepiece as spm

class Tokenizer:
    def __init__(self, model_path):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model_path))
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text, out_type=int)

    def decode(self, token_ids):
        return self.sp.decode(token_ids)
