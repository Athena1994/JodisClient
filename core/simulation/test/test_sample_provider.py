from unittest import TestCase

from core.data.data_provider import ChunkType, Sample
from core.simulation.sample_provider import SampleProvider
from core.simulation.test.mockup import DummyChunkProvider, DummyContProvider


class TestSampleProvider(TestCase):

    def test_sample_provider(self):

        try:
            sp = SampleProvider({})  # noqa
            self.fail("Expected ValueError")
        except ValueError:
            pass

        cont1 = DummyContProvider(self)
        cont2 = DummyContProvider(self)
        chunk1 = DummyChunkProvider(self)
        chunk2 = DummyChunkProvider(self)

        # assert differing signatures are not accepted
        chunk1.chunk_signature = "sig1"
        chunk2.chunk_signature = "sig2"
        try:
            sp = SampleProvider({  # noqa
                "cont1": cont1,
                "cont2": cont2,
                "chunk1": chunk1,
                "chunk2": chunk2
            })
            self.fail("Expected ValueError")
        except ValueError:
            pass

        chunk2.chunk_signature = "sig1"
        sp = SampleProvider({
            "cont1": cont1,
            "cont2": cont2,
            "chunk1": chunk1,
            "chunk2": chunk2
        })

        # assert exception on invalid state
        try:
            sp.get_next_samples()
            self.fail("Expected RuntimeError")
        except RuntimeError:
            pass
        try:
            sp.current_episode_complete()()
            self.fail("Expected RuntimeError")
        except RuntimeError:
            pass
        try:
            sp.update_values(None)
            self.fail("Expected RuntimeError")
        except RuntimeError:
            pass

        # start session
        sp.reset(ChunkType.VALIDATION)
        self.assertEqual(cont1.last_iter_type, ChunkType.VALIDATION)
        self.assertEqual(cont2.last_iter_type, ChunkType.VALIDATION)
        self.assertEqual(chunk1.last_iter_type, ChunkType.VALIDATION)
        self.assertEqual(chunk2.last_iter_type, ChunkType.VALIDATION)
        sp.reset(ChunkType.TEST)
        self.assertEqual(cont1.last_iter_type, ChunkType.TEST)
        self.assertEqual(cont2.last_iter_type, ChunkType.TEST)
        self.assertEqual(chunk1.last_iter_type, ChunkType.TEST)
        self.assertEqual(chunk2.last_iter_type, ChunkType.TEST)
        sp.reset(ChunkType.TRAINING)
        self.assertEqual(cont1.last_iter_type, ChunkType.TRAINING)
        self.assertEqual(cont2.last_iter_type, ChunkType.TRAINING)
        self.assertEqual(chunk1.last_iter_type, ChunkType.TRAINING)
        self.assertEqual(chunk2.last_iter_type, ChunkType.TRAINING)
        chunk1.assert_chunk_change()
        chunk2.assert_chunk_change()

        # assert current_episode_complete behaviour
        self.assertFalse(sp.current_episode_complete())
        chunk1.reader_exhausted = True
        self.assertTrue(sp.current_episode_complete())
        chunk1.reader_exhausted = False
        self.assertFalse(sp.current_episode_complete())

        # assert get_next_samples behaviour
        out_cont1 = {'a': 1}
        out_cont2 = {'b': 2}
        out_chunk1 = {'c': 3}
        out_chunk2 = {'d': 4}

        cont1.next_sample = Sample(None, out_cont1)
        cont2.next_sample = Sample(None, out_cont2)
        chunk1.next_sample = Sample(None, out_chunk1)
        chunk2.next_sample = Sample(None, out_chunk2)

        self.assertDictEqual(sp.get_next_samples(), {
            "cont1": cont1.next_sample,
            "cont2": cont2.next_sample,
            "chunk1": chunk1.next_sample,
            "chunk2": chunk2.next_sample
        })

        out_cont1['a'] += 1
        out_cont2['b'] += 1
        out_chunk1['c'] += 1
        out_chunk2['d'] += 1
        s = sp.get_next_samples()
        self.assertEqual(s['cont1'].context['a'], 2)
        self.assertEqual(s['cont2'].context['b'], 3)
        self.assertEqual(s['chunk1'].context['c'], 4)
        self.assertEqual(s['chunk2'].context['d'], 5)

        # assert update_values behaviour
        out_cont1['a'] += 1
        out_cont2['b'] += 1
        out_chunk1['c'] += 1
        out_chunk2['d'] += 1
        cont1._expected_update_sample = s["cont1"]
        cont2._expected_update_sample = s["cont2"]

        sp.update_values(s)
        self.assertEqual(s['cont1'].context['a'], 3)
        self.assertEqual(s['cont2'].context['b'], 4)
        self.assertEqual(s['chunk1'].context['c'], 4)
        self.assertEqual(s['chunk2'].context['d'], 5)

        # assert chunk change behaviour
        chunk1.assert_no_chunk_change()
        chunk2.assert_no_chunk_change()

        chunk1.reader_exhausted = True
        self.assertIsNotNone(sp.get_next_samples())
        self.assertFalse(chunk1.reader_exhausted)
        chunk1.assert_chunk_change()
        chunk2.assert_chunk_change()

        chunk1.last_chunk_reached = True
        chunk1.reader_exhausted = True
        try:
            sp.get_next_samples()
            self.fail("Expected StopIteration")
        except StopIteration:
            pass
