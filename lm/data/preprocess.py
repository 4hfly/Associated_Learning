from tokenizers.trainers import BpeTrainer
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


# We initialize our trainer, giving him the details about the vocabulary we want to generate
trainer = BpeTrainer(special_tokens=["[PAD]", "[UNK]", "[SEP]"], vocab_size=25000,
                     show_progress=True, initial_alphabet=ByteLevel.alphabet())
tokenizer.train(files=["pretrain/news.2012.en.shuffled", "train/en-corpus/commoncrawl.fr-en.en", "train/en-corpus/europarl-v7.fr-en.en",
                       "train/en-corpus/giga-fren.release2.fixed.en", "train/en-corpus/news-commentary-v9.fr-en.en", "train/en-corpus/undoc.2000.fr-en.en"], trainer=trainer)

print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))


tokenizer.model.save('.')

# Let's tokenizer a simple input
tokenizer.model = BPE('vocab.json', 'merges.txt')
encoding = tokenizer.encode("This is a simple input to be tokenized [PAD]")

print("Encoded string: {}".format(encoding.tokens))
print('encoding ids', encoding.ids)
decoded = tokenizer.decode(encoding.ids)
print("Decoded string: {}".format(decoded))
