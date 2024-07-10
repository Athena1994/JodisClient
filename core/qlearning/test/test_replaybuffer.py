import unittest

from core.qlearning.replay_buffer import Experience, ReplayBuffer

class TestReplayBuffer(unittest.TestCase):

    def test_multidim_sample(self):
        rb = ReplayBuffer(128)
        for i in range(128):
            rb.add_experience(Experience([i], i, i, [i+1]), 1)
        res = rb.sample_experiences((16, 64), True)['prev_state']
        self.assertEqual((16, 64), res.shape)

    def test_reward_is_properly_processed(self):
        rb = ReplayBuffer(64)
        rb.add_experience(Experience(0, 0, .42, 0), 1)
        self.assertEqual(rb.get_experiences()[0].reward, .42)

    def test_add_experiences(self):
        def sort(ar):
            return sorted(ar, key=lambda e: e[1])

        test_size = 640
        replay_buffer = ReplayBuffer(test_size)

        #
        # add 2*TEST_SIZE entries and assert resulting buffer after each add
        #
        check_list = []
        for i in range(2 * test_size):
            new_exp = Experience({'value': i}, i, 0, {'value': i + 1})

            # update comparison list
            check_list += [new_exp]
            if len(check_list) > test_size:
                del check_list[0]

            replay_buffer.add_experience(new_exp, 1)
            result = replay_buffer.get_experiences()
            self.assertListEqual(sort(check_list), sort(result))
            self.assertEqual(len(check_list), len(replay_buffer))

        #
        # assert that weights are respected while sampling
        #
        sample_num = 10000
        replay_buffer.clear()
        vals = [{'val': 1, 'weight': 5},
                {'val': 2, 'weight': 10},
                {'val': 3, 'weight': 20},
                {'val': 4, 'weight': 7}]
        total_weight = sum(map(lambda e: e['weight'], vals))
        for v in vals:
            new_exp = Experience((0,), v['val'], v['weight'], (1,))
            replay_buffer.add_experience(new_exp, new_exp.reward)
        experiences = replay_buffer.sample_experiences(sample_num, True)
        samples = experiences['action']

        for v in vals:
            num_with_value = samples[samples == v['val']].shape[0]
            frequency = num_with_value / samples.shape[0]
            self.assertAlmostEqual(frequency, v['weight']/total_weight, 1)

        # assert that weights are respected while sampling without replace
        buffer_size = sample_num * 10
        replay_buffer = ReplayBuffer(buffer_size)
        for v in vals:
            for _ in range(buffer_size//len(vals)):
                replay_buffer.add_experience(Experience(0, v['val'], 0, 0),
                                             v['weight'])

        experiences = replay_buffer.sample_experiences(sample_num, False)
        samples = experiences['action']
        for v in vals:
            num_with_value = samples[samples == v['val']].shape[0]
            frequency = num_with_value / samples.shape[0]
            self.assertAlmostEqual(frequency, v['weight']/total_weight, 1)

        experiences = replay_buffer.sample_experiences(buffer_size, False)
        samples = experiences['action']
        for v in vals:
            num_with_value = samples[samples == v['val']].shape[0]
            frequency = num_with_value / samples.shape[0]
            self.assertAlmostEqual(frequency, 1/len(vals), 1)

