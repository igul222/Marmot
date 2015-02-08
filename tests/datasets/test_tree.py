import unittest
import numpy
import theano

from marmot.datasets.trees import Trees

class TreeTest(unittest.TestCase):

    def test_parse(self):
        node = Trees.parse("(1 (2 (0 my) (0 name)) (3 (0 is) (0 ishaan)))")
        self.assertEqual(node.label, 1)
        self.assertEqual(node.left.label, 2)
        self.assertEqual(node.right.label, 3)
        self.assertEqual(node.left.left.word, "my")
        self.assertEqual(node.left.right.word, "name")
        self.assertEqual(node.right.left.word, "is")
        self.assertEqual(node.right.right.word, "ishaan")

        self.assertFalse(node.is_leaf())
        self.assertTrue(node.left.left.is_leaf())

    def test_node_traverse(self):
        tree = Trees.parse("(1 (2 (0 my) (0 name)) (3 (0 is) (0 ishaan)))")

        def get_word(node):
            if node.is_leaf():
                return node.word
        words = tree.traverse(get_word)

        self.assertEqual(
            words,
            ["my","name","is","ishaan"]
        )

    def test_build_wordmap(self):
        tree = Trees.parse("(1 (2 (0 my) (0 name)) (3 (0 is) (0 name)))")
        wordmap = Trees._build_wordmap([tree])
        self.assertEqual(
            wordmap,
            {Trees.UNK: 0, 'my': 1, 'name': 2, 'is': 3}
        )

    def test_is_leafs(self):
        tree = Trees.parse("(1 (2 (0 my) (0 name)) (3 (0 is) (0 ishaan)))")
        self.assertEqual(
            Trees._is_leafs([tree]),
            [[1,1,0,1,1,0,0]]
        )

    def test_word_indices(self):
        tree = Trees.parse("(1 (2 (0 my) (0 name)) (3 (0 is) (0 ishaan)))")
        wordmap = Trees._build_wordmap([tree])
        self.assertEqual(
            Trees._word_indices([tree], wordmap),
            [[1,2,0,3,4,0,0]]
        )

    def test_word_indices_unk(self):
        tree = Trees.parse("(1 (2 (0 my) (0 name)) (3 (0 is) (0 ishaan)))")
        wordmap = Trees._build_wordmap([tree])
        tree2 = Trees.parse("(0 (0 ishaan) (0 rocks))")
        self.assertEqual(
            Trees._word_indices([tree2], wordmap),
            [[4,0,0]]
        )


    def test_child_indices(self):
        tree = Trees.parse("(1 (2 (0 my) (0 name)) (3 (0 is) (0 ishaan)))")
        self.assertEqual(
            Trees._child_indices([tree]),
            [[[0,0],
              [0,0],
              [0,1],
              [0,0],
              [0,0],
              [3,4],
              [2,5]]]
        )

    def test_targets(self):
        tree = Trees.parse("(1 (2 (0 my) (0 name)) (3 (0 is) (0 ishaan)))")
        self.assertEqual(
            Trees._targets([tree]),
            [[0,0,2,0,0,3,1]]
        )
