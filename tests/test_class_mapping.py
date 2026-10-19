import unittest

from class_mapping import canonical_class_name, sorted_class_names


class ClassMappingTests(unittest.TestCase):
    def test_non_card_merge(self):
        self.assertEqual(canonical_class_name("0_nocard"), "0_nocard")
        self.assertEqual(canonical_class_name("54_zerocard"), "0_nocard")
        self.assertEqual(canonical_class_name("54"), "0_nocard")

    def test_numeric_order(self):
        self.assertEqual(sorted_class_names(["10", "2", "0_nocard", "1"]), ["0_nocard", "1", "2", "10"])


if __name__ == "__main__":
    unittest.main()
