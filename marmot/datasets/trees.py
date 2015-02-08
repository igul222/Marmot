import numpy
import theano
import theano.tensor as T

import helpers

class Node(object):
    """Node in a parse tree."""

    def __init__(self):
        self.label = None
        self.parent = None

    def traverse(self, fn):
        """Traverse the subtree rooted at this node bottom-up, calling fn(node) 
           at each step and returning an array of all values returned by fn."""
        raise NotImplementedError

    def is_leaf(self):
        """Return true iff this node is a leaf node."""
        return isinstance(self, LeafNode)

class InternalNode(Node):
    """Internal node; has left/right children."""

    def __init__(self):
        super(InternalNode, self).__init__()
        self.left = None
        self.right = None

    def __repr__(self):
        return "({0} {1} {2})".format(self.label, self.left, self.right)

    def traverse(self, fn):
        results = []
        results.extend(self.left.traverse(fn))
        results.extend(self.right.traverse(fn))
        results.append(fn(self))
        return [x for x in results if x is not None]

class LeafNode(Node):
    """Leaf node; has a word."""

    def __init__(self):
        super(LeafNode, self).__init__()
        self.word = None

    def __repr__(self):
        return "({0} {1})".format(self.label, self.word)

    def traverse(self, fn):
        return [fn(self)]

class TreeMinibatch(object):
    def __init__(self, lengths, is_leafs, word_indices, child_indices, targets):
        self.lengths = lengths
        self.is_leafs = is_leafs
        self.word_indices = word_indices
        self.child_indices = child_indices
        self.targets = targets

class Trees(object):
    """Dataset class for trees (e.g. semantic parse trees for RNTN).

       theano.scan doesn't support recursion, so we traverse each tree up front 
       in the order the network would, generating flat sequences of operations 
       that can be more easily handled by theano.scan."""

    # Unknown word token
    UNK = 'UNK'

    def __init__(
        self,
        trees,
        minibatch_size=128,
        wordmap=None,
        shuffle=True
        ):

        self.wordmap = wordmap or self._build_wordmap(trees)

        is_leafs = self._is_leafs(trees)
        word_indices = self._word_indices(trees, self.wordmap)
        child_indices = self._child_indices(trees)
        targets = self._targets(trees)

        # Apply padding
        is_leafs, lengths = helpers.pad(is_leafs, 0)
        word_indices, _ = helpers.pad(word_indices, 0)
        child_indices, _ = helpers.pad(child_indices, 0)
        targets, _ = helpers.pad(targets, 0)

        if shuffle:
            # Shuffle
            lengths, \
            is_leafs, \
            word_indices, \
            child_indices, \
            targets = helpers.parallel_shuffle(
                (lengths, is_leafs, word_indices, child_indices, targets)
            )

            # Now sort in order of increasing input length
            lengths, \
            is_leafs, \
            word_indices, \
            child_indices, \
            targets = helpers.sort_by(
                lengths,
                (lengths, is_leafs, word_indices, child_indices, targets)
            )

        # Transpose data from example -> sequence to sequence -> example
        is_leafs = is_leafs.transpose(1, 0)
        word_indices = word_indices.transpose(1, 0)
        child_indices = child_indices.transpose(1, 2, 0) # seq -> l/r -> ex
        targets = targets.transpose(1, 0)

        # Load into theano shared vars
        self.is_leafs = theano.shared(is_leafs, borrow=True)
        self.word_indices = theano.shared(word_indices, borrow=True)
        self.child_indices = theano.shared(child_indices, borrow=True)
        self.targets = theano.shared(targets, borrow=True)
        # these need to be stored on the GPU as floats and casted when needed
        self.lengths = theano.shared(
            lengths.astype(theano.config.floatX), 
            borrow=True)

        self.minibatch_size = minibatch_size
        self.minibatch_count = int(numpy.ceil(float(len(trees)) / minibatch_size))

    def minibatch(self, index):
        mb_start = self.minibatch_size * index
        mb_end = T.min((self.minibatch_size * (index + 1), self.is_leafs.shape[1]), axis=0)

        lengths = T.cast(self.lengths[mb_start:mb_end], 'int32')
        # Explicitly specify axis to work around Theano bug
        mb_max_len = T.max(lengths, axis=0)

        is_leafs = self.is_leafs[:mb_max_len, mb_start:mb_end]
        word_indices = self.word_indices[:mb_max_len, mb_start:mb_end]
        child_indices = self.child_indices[:mb_max_len, :, mb_start:mb_end]
        targets = self.targets[:mb_max_len, mb_start:mb_end]

        return TreeMinibatch(
            lengths,
            is_leafs,
            word_indices,
            child_indices,
            targets
        )

    @classmethod
    def parse(cls, string, parent=None):
        """Create a tree (recursively) from a string with format:
           (label [word | leftchild rightchild])"""

        label = int(string[1])

        string = string[3:-1] # strip everything but "word/leftchild rightchild"

        if not ' ' in string:
            node = LeafNode()
            node.word = string
        else:
            # find the split between left and right children and recurse.
            pos = 0
            depth = 0
            while depth > 0 or string[pos] != ' ':
                if string[pos] == '(':
                    depth += 1
                elif string[pos] == ')':
                    depth -= 1
                pos += 1

            node = InternalNode()
            node.left = cls.parse(string[:pos], node)
            node.right = cls.parse(string[pos+1:], node)

        node.label = label
        node.parent = parent
        return node

    @classmethod
    def _build_wordmap(cls, trees):
        """Build a wordmap (a dictionary of words -> integers) from a given 
           array of trees."""

        wordmap = {cls.UNK: 0}

        def add_word(node):
            if node.is_leaf() and node.word not in wordmap:
                wordmap[node.word] = len(wordmap)

        for tree in trees:
            tree.traverse(add_word)

        return wordmap

    @staticmethod
    def _is_leafs(trees):
        """For each node in each tree, 1.0 if the node is a leaf, and 0.0 
           otherwise."""
        def is_leaf(node):
            return numpy.float32(isinstance(node, LeafNode))
        return [tree.traverse(is_leaf) for tree in trees]

    @classmethod
    def _word_indices(cls, trees, wordmap):
        """For each node in each tree, the index in the wordmap of the given
           node."""

        def word_index(node):
            if isinstance(node, LeafNode):
                if node.word in wordmap:
                    return numpy.float32(wordmap[node.word])
                else:
                    return numpy.float32(wordmap[cls.UNK])
            else:
                return numpy.float32(0)

        return [tree.traverse(word_index) for tree in trees]

    @staticmethod
    def _child_indices(trees):
        """For each node in each tree, an array of the traverse-order indices of
           the left and right children of the given node."""

        def tree_child_indices(tree):
            counter = [0] # hack to work around Python's weak closure support
            indices = {}
            def child_indices(node):
                indices[node] = numpy.float32(counter[0])
                counter[0] += 1
                if isinstance(node, LeafNode):
                    return [numpy.float32(0), numpy.float32(0)]
                else:
                    return [indices[node.left], indices[node.right]]
            return tree.traverse(child_indices)

        return [tree_child_indices(tree) for tree in trees]

    @staticmethod
    def _targets(trees):
        """For each node in each tree, the target class for that node."""
        def label(node):
            return numpy.float32(node.label)
        return [tree.traverse(label) for tree in trees]
