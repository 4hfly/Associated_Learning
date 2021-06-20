from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel


tokenizer = Tokenizer(BPE())

tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase()
])

# Our tokenizer also needs a pre-tokenizer responsible for converting the input to a ByteLevel representation.
tokenizer.pre_tokenizer = ByteLevel()

# And finally, let's plug a decoder so we can recover from a tokenized input to the original one
tokenizer.decoder = ByteLevelDecoder()

from tokenizers.trainers import BpeTrainer

# We initialize our trainer, giving him the details about the vocabulary we want to generate
trainer = BpeTrainer(special_tokens=["[PAD]","[UNK]","[SEP]"],vocab_size=25000, show_progress=True, initial_alphabet=ByteLevel.alphabet())
tokenizer.train(files=["pretrain/news.2012.fr.shuffled", "train/fr-corpus/commoncrawl.fr-en.fr", "train/fr-corpus/europarl-v7.fr-en.fr", "train/fr-corpus/giga-fren.release2.fixed.fr", "train/fr-corpus/news-commentary-v9.fr-en.fr", "train/fr-corpus/undoc.2000.fr-en.fr"], trainer=trainer)

print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))


tokenizer.model.save('./fr/')

# Let's tokenizer a simple input
tokenizer.model = BPE('vocab.json', 'merges.txt')
encoding = tokenizer.encode("This is a simple input to be tokenized [PAD]")

print("Encoded string: {}".format(encoding.tokens))
print('encoding ids',encoding.ids)
decoded = tokenizer.decode(encoding.ids)
print("Decoded string: {}".format(decoded))

