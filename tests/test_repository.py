import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DICTIONARY_SHA256 = (
    "b5388af5425af0596768b0b72a531a5f71566fe0729325f78913e69f3bf55a6e"
)


class RepositoryTests(unittest.TestCase):
    def test_original_dictionary_is_preserved(self):
        path = ROOT / "data" / "dictionary.json"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), ORIGINAL_DICTIONARY_SHA256)
        dictionary = json.loads(path.read_text())
        self.assertEqual(len(dictionary), 256)
        self.assertEqual(len({concept["name"] for concept in dictionary}), 256)

    def test_reproduction_entrypoints_exist(self):
        expected = [
            "scripts/data/mine_attributes.py",
            "scripts/data/caption_images.py",
            "scripts/data/compute_labels.py",
            "scripts/dictionary/build_dictionary.py",
            "scripts/dictionary/generate_contrastive_prompts.py",
            "scripts/dictionary/verify_concepts.py",
            "scripts/dictionary/build_directions.py",
            "scripts/visualization/make_playground_directions.py",
        ]
        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())


if __name__ == "__main__":
    unittest.main()
