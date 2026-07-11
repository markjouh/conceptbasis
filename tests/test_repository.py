import json
import hashlib
from pathlib import Path
import unittest

import pandas as pd

from conceptbasis.splits import image_class


ROOT = Path(__file__).resolve().parents[1]
class RepositoryTests(unittest.TestCase):
    def test_dictionary_is_well_formed(self):
        path = ROOT / "data" / "dictionary.json"
        dictionary = json.loads(path.read_text())
        self.assertEqual(len(dictionary), 256)
        self.assertEqual(len({concept["name"] for concept in dictionary}), 256)
        self.assertTrue(all(concept["members"] for concept in dictionary))
        provenance = json.loads((ROOT / "data" / "dictionary_provenance.json").read_text())
        self.assertEqual(
            provenance["dictionary_sha256"],
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        self.assertEqual(provenance["attribute_split"], "train")

    def test_reproduction_entrypoints_exist(self):
        expected = [
            "scripts/data/make_class_splits.py",
            "scripts/data/partition_attributes.py",
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

    def test_class_split_covers_full_and_cc0_sets(self):
        manifest = json.loads((ROOT / "data" / "splits.json").read_text())
        image_ids = json.loads((ROOT / "data" / "image_ids.json").read_text())
        full_classes = {image_class(image_id) for image_id in image_ids}
        self.assertEqual(full_classes, set(manifest["classes"]))
        self.assertEqual(set(manifest["classes"].values()), {"train", "dev", "test"})

        for split in ("train", "dev"):
            rows = [
                json.loads(line)
                for line in (ROOT / "data" / f"attributes_{split}.jsonl").read_text().splitlines()
                if line
            ]
            self.assertTrue(rows)
            self.assertEqual(
                {manifest["classes"][image_class(row["image_id"])] for row in rows},
                {split},
            )

    @unittest.skipUnless((ROOT / "data" / "labels.parquet").exists(), "local labels cache")
    def test_label_rows_follow_class_split(self):
        manifest = json.loads((ROOT / "data" / "splits.json").read_text())
        labels = pd.read_parquet(ROOT / "data" / "labels.parquet")
        expected = [manifest["classes"][concept] for concept in labels.concept]
        self.assertEqual(list(labels.split), expected)
        self.assertTrue((labels.groupby("concept").split.nunique() == 1).all())


if __name__ == "__main__":
    unittest.main()
