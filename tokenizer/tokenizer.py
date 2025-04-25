from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.models import Unigram
from tokenizers.trainers import UnigramTrainer



class CustomTokenizer:
    def __init__(self,model_type='bpe'):
        self.model_type=model_type.lower()

        if self.model_type == 'bpe':
            self.tokenizer=Tokenizer(BPE())
            self.trainer=BpeTrainer(vocab_size=50000,min_frequency=2,show_progress=True)
        elif self.model_type == 'wordpiece':
            self.tokenizer=Tokenizer(WordPiece())
            self.trainer=WordPieceTrainer(vocab_size=50000,min_frequency=2,show_progress=True)
        elif self.model_type == 'unigram':
            self.tokenizer=Tokenizer(Unigram())
            self.trainer=UnigramTrainer(vocab_size=50000,min_frequency=2,show_progress=True)
        else:
            raise ValueError(f"Unsupport model type: {self.model_type}")
        
        self.tokenizer.pre_tokenizer=Whitespace()

    def train(self,files):
        self.tokenizer.train(files,self.trainer)
    
    def save(self,model_path):
        self.tokenizer.save(model_path)
    
    def encode(self,text):
        return self.tokenizer.encode(text)
    
    def decode(self,token_ids):
        return self.tokenizer.decode(token_ids)
    
    def load(self,model_path):
        self.tokenizer=Tokenizer.from_file(model_path)