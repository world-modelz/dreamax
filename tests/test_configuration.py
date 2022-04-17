import unittest
import glob
import json

from dreamer.configuration import Dataclass, DreamerConfiguration


class TestConfiguration(unittest.TestCase):
    def test_read(self):

        class TestA(Dataclass):
            a: int = 5
            b: list = [7, 7]

        class TestB(TestA):
            a: int = 33
            x: TestA = TestA()

        # Init configs with default params.
        test_a = TestA()
        test_b = TestB()

        # Test default params in configs.
        self.assertEqual(test_a.a, 5)
        self.assertListEqual(test_a.b, [7, 7])

        self.assertEqual(test_b.a, 33)
        self.assertListEqual(test_a.b, [7, 7])
        self.assertEqual(test_b.x.a, 5)
        self.assertListEqual(test_b.x.b, [7, 7])

        # Change some params.
        new_config = {'a': 77, 'b': [2, 3], 'x': {'a': 1111}}
        test_b.update(new_config)

        self.assertEqual(test_b.a, 77)
        self.assertListEqual(test_b.b, [2, 3])
        self.assertEqual(test_b.x.a, 1111)
        self.assertListEqual(test_b.x.b, [7, 7])

        # Change some params.
        new_config = {'x': {'a': 7, 'b': [1, 2, 3, 4, 5, 6]}}
        test_b.update(new_config)

        self.assertEqual(test_b.a, 77)
        self.assertListEqual(test_b.b, [2, 3])
        self.assertEqual(test_b.x.a, 7)
        self.assertListEqual(test_b.x.b, [1, 2, 3, 4, 5, 6])

    def test_load_default_configs(self):

        config_paths = glob.glob('../configs/*.json')

        for config_path in config_paths:
            config = DreamerConfiguration(
                json.load(open(config_path, 'r')), load_with_warning=False)
            del config
