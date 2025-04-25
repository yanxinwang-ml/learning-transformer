import transformers
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from tokenizer import CustomTokenizer
model_type='bpe'
tokenizer=CustomTokenizer(model_type=model_type)
tokenizer.train(files=['parallel_corpus.txt'])
tokenizer.save(f"{model_type}_tokenizer.json")
encoded=tokenizer.encode('how are you')
print(encoded.tokens)
decoded=tokenizer.decode(encoded.ids)
print(decoded)