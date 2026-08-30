import unittest

from memory_reader import clean_name


class MemoryReaderDisplayNameTests(unittest.TestCase):
    def test_asset_proven_internal_role_names(self):
        expected = {
            "Baron": "Chancellor",
            "Cipher": "Witch",
            "Imp": "Baa",
            "Illuzionist": "Shaman",
            "Judge2": "Judge",
            "Marionette": "Twin Minion",
            "Mezepheles": "Puppeteer",
            "Puzzlemaster": "Plague Doctor",
            "Skinwalker": "Mutant",
            # Spy is a distinct native Minion implementation, not the managed
            # class bound by the public Plague Doctor CharacterData.
            "Spy": "Spy",
            # Arbiter is a distinct, currently unbound native role.
            "Arbiter": "Arbiter",
        }

        for internal_name, display_name in expected.items():
            with self.subTest(internal_name=internal_name):
                self.assertEqual(clean_name(internal_name), display_name)

    def test_asset_proven_names_with_numeric_suffixes(self):
        expected = {
            "Baron_04539999": "Chancellor",
            "Imp_58992273": "Baa",
            "Judge_87202475": "Judge",
            "Marionette_21595": "Twin Minion",
            "Mezepheles_09511163": "Puppeteer",
            "Plague Doctor_49312486": "Plague Doctor",
            "Shaman_26945607": "Shaman",
            "Witch_25286521": "Witch",
        }

        for native_name, display_name in expected.items():
            with self.subTest(native_name=native_name):
                self.assertEqual(clean_name(native_name), display_name)

    def test_public_saint_names_are_not_bombardier_aliases(self):
        self.assertEqual(clean_name("Saint"), "Saint")
        self.assertEqual(clean_name("SaintVillager"), "SaintVillager")
        self.assertEqual(clean_name("Saint_12345"), "Saint")


if __name__ == "__main__":
    unittest.main()
