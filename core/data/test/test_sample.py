from unittest import TestCase

from torch import Tensor
from core.data.data_provider import Sample


class TestSample(TestCase):
    def test_sample(self):
        s_none_a1   = Sample(None, {'a': 1}) # noqa
        s_none_a1_2 = Sample(None, {'a': 1})

        s_t1_a1   = Sample(Tensor(range(5)), {'a': 1}) # noqa
        s_t1_a1_2 = Sample(Tensor(range(5)), {'a': 1})

        s_t2_a1   = Sample(Tensor(range(5, 10)), {'a': 1}) # noqa
        s_t2_a1_2 = Sample(Tensor(range(5, 10)), {'a': 1})

        s_t1_none   = Sample(Tensor(range(5)), None) # noqa
        s_t1_none_2 = Sample(Tensor(range(5)), None)

        s_t2_a1   = Sample(Tensor(range(5, 10)), {'a': 1}) # noqa
        s_t2_a1_2 = Sample(Tensor(range(5, 10)), {'a': 1})

        s_t2_none   = Sample(Tensor(range(5, 10)), None) # noqa
        s_t2_none_2 = Sample(Tensor(range(5, 10)), None)

        s_none_a2   = Sample(None, {'a': 2}) # noqa
        s_none_a2_2 = Sample(None, {'a': 2})

        s_t1_a2   = Sample(Tensor(range(5)), {'a': 2}) # noqa
        s_t1_a2_2 = Sample(Tensor(range(5)), {'a': 2})

        s_t2_a2   = Sample(Tensor(range(5, 10)), {'a': 2}) # noqa
        s_t2_a2_2 = Sample(Tensor(range(5, 10)), {'a': 2})

        self.assertEqual(s_none_a1, s_none_a1_2)
        self.assertEqual(s_t1_a1, s_t1_a1_2)
        self.assertEqual(s_t2_a1, s_t2_a1_2)
        self.assertEqual(s_none_a2, s_none_a2_2)
        self.assertEqual(s_t1_a2, s_t1_a2_2)
        self.assertEqual(s_t2_a2, s_t2_a2_2)
        self.assertEqual(s_t1_none, s_t1_none_2)
        self.assertEqual(s_t2_none, s_t2_none_2)

        self.assertEqual(hash(s_none_a1), hash(s_none_a1_2))
        self.assertEqual(hash(s_none_a2), hash(s_none_a2_2))
        self.assertEqual(hash(s_t1_a1), hash(s_t1_a1_2))
        self.assertEqual(hash(s_t1_a2), hash(s_t1_a2_2))
        self.assertEqual(hash(s_t1_none), hash(s_t1_none_2))
        self.assertEqual(hash(s_t2_a1), hash(s_t2_a1_2))
        self.assertEqual(hash(s_t2_a2), hash(s_t2_a2_2))
        self.assertEqual(hash(s_t2_none), hash(s_t2_none_2))

        self.assertNotEqual(s_none_a1, s_none_a2)
        self.assertNotEqual(s_none_a1, s_t1_a1)
        self.assertNotEqual(s_none_a1, s_t1_a2)
        self.assertNotEqual(s_none_a1, s_t1_none)
        self.assertNotEqual(s_none_a1, s_t2_a1)
        self.assertNotEqual(s_none_a1, s_t2_a2)
        self.assertNotEqual(s_none_a1, s_t2_none)

        self.assertNotEqual(s_none_a2, s_none_a1)
        self.assertNotEqual(s_none_a2, s_t1_a1)
        self.assertNotEqual(s_none_a2, s_t1_a2)
        self.assertNotEqual(s_none_a2, s_t1_none)
        self.assertNotEqual(s_none_a2, s_t2_a1)
        self.assertNotEqual(s_none_a2, s_t2_a2)
        self.assertNotEqual(s_none_a2, s_t2_none)

        self.assertNotEqual(s_t1_a1, s_none_a1)
        self.assertNotEqual(s_t1_a1, s_none_a2)
        self.assertNotEqual(s_t1_a1, s_t1_a2)
        self.assertNotEqual(s_t1_a1, s_t1_none)
        self.assertNotEqual(s_t1_a1, s_t2_a1)
        self.assertNotEqual(s_t1_a1, s_t2_a2)
        self.assertNotEqual(s_t1_a1, s_t2_none)

        self.assertNotEqual(s_t1_a2, s_none_a1)
        self.assertNotEqual(s_t1_a2, s_none_a2)
        self.assertNotEqual(s_t1_a2, s_t1_a1)
        self.assertNotEqual(s_t1_a2, s_t1_none)
        self.assertNotEqual(s_t1_a2, s_t2_a1)
        self.assertNotEqual(s_t1_a2, s_t2_a2)
        self.assertNotEqual(s_t1_a2, s_t2_none)

        self.assertNotEqual(s_t1_none, s_none_a1)
        self.assertNotEqual(s_t1_none, s_none_a2)
        self.assertNotEqual(s_t1_none, s_t1_a1)
        self.assertNotEqual(s_t1_none, s_t1_a2)
        self.assertNotEqual(s_t1_none, s_t2_a1)
        self.assertNotEqual(s_t1_none, s_t2_a2)
        self.assertNotEqual(s_t1_none, s_t2_none)

        self.assertNotEqual(s_t2_a1, s_none_a1)
        self.assertNotEqual(s_t2_a1, s_none_a2)
        self.assertNotEqual(s_t2_a1, s_t1_a1)
        self.assertNotEqual(s_t2_a1, s_t1_a2)
        self.assertNotEqual(s_t2_a1, s_t1_none)
        self.assertNotEqual(s_t2_a1, s_t2_a2)
        self.assertNotEqual(s_t2_a1, s_t2_none)

        self.assertNotEqual(s_t2_a2, s_none_a1)
        self.assertNotEqual(s_t2_a2, s_none_a2)
        self.assertNotEqual(s_t2_a2, s_t1_a1)
        self.assertNotEqual(s_t2_a2, s_t1_a2)
        self.assertNotEqual(s_t2_a2, s_t1_none)
        self.assertNotEqual(s_t2_a2, s_t2_a1)
        self.assertNotEqual(s_t2_a2, s_t2_none)

        self.assertNotEqual(s_t2_none, s_none_a1)
        self.assertNotEqual(s_t2_none, s_none_a2)
        self.assertNotEqual(s_t2_none, s_t1_a1)
        self.assertNotEqual(s_t2_none, s_t1_a2)
        self.assertNotEqual(s_t2_none, s_t1_none)
        self.assertNotEqual(s_t2_none, s_t2_a1)
        self.assertNotEqual(s_t2_none, s_t2_a2)
