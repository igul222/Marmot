import unittest
import marmot
import numpy

class LayerTest(unittest.TestCase):

    def test_dump_and_load_params(self):

        def make_model():
            model = marmot.layers.Input(10)
            return marmot.layers.Feedforward(model, 5)

        m1 = make_model()
        m2 = make_model()

        dataset = marmot.datasets.Simple([numpy.random.randn(10)], [0])

        self.assertFalse(
            numpy.array_equal(
                m1.activations(dataset).eval(),
                m2.activations(dataset).eval()
            )
        )

        m2.load_params(m1.dump_params())

        numpy.testing.assert_array_equal(
            m1.activations(dataset).eval(),
            m2.activations(dataset).eval()
        )
