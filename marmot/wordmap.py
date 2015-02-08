import numpy
import theano

class Wordmap(object):
    """A two-way mapping between words and indices."""

    UNK = '[UNK]'

    def __init__(self, words=None, strings=None, use_unk=False):
        self.indices = {}
        self.words = {}
        self._len = 0
        self._use_unk = use_unk
        if use_unk:
            self.add_words([self.UNK])
        if words:
            self.add_words(words)
        if strings:
            self.add_strings(strings)

    def __len__(self):
        return self._len

    def index_for_word(self, word):
        """Get the index corresponding to a given word."""
        if word in self.indices:
            return self.indices[word]
        elif self._use_unk:
            return self.indices[self.UNK]
        else:
            raise KeyError(word)

    def word_for_index(self, index):
        """Get the word corresponding to a given index."""
        return self.words[index]

    def add_words(self, words):
        """Add each word in words to the wordmap."""
        for word in words:
            if word not in self.indices:
                self.indices[word] = self._len
                self.words[self._len] = word
                self._len += 1

    def add_strings(self, strings):
        """Add each word in each string to the wordmap."""
        for string in strings:
            words = string.split(' ')
            self.add_words(words)

    def to_wordbags(self, strings):
        """Convert each string in strings to a bag-of-words vector using this 
           wordmap. The output vectors are normalized for sentence length."""

        def bagify(string):
            bag = numpy.zeros((len(self),), dtype=theano.config.floatX)
            words = string.split(' ')
            for word in words:
                if word in self.indices:
                    bag[self.indices[word]] += 1.0
                elif self._use_unk:
                    bag[self.indices[self.UNK]] += 1.0
            return bag / len(words)

        return numpy.array(map(bagify, strings))