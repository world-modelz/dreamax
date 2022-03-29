import unittest
from dreamer.configuartion import load_configuration_file


class TestConfiguration(unittest.TestCase):
    def test_read(self):
        cfg = load_configuration_file('dreamer/config.json')
        
        default_config = cfg['defaults']
        self.assertTrue(default_config.jit)
        
        debug_config = cfg['debug']
        self.assertFalse(debug_config.jit)
        self.assertEqual(debug_config.time_limit, 100)
        self.assertEqual(debug_config.training_steps_per_epoch, 200)
        self.assertEqual(debug_config.replay.capacity, 10)
        self.assertEqual(debug_config.replay.batch, 10)
        self.assertEqual(debug_config.replay.sequence_length, 8)
