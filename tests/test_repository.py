import json
import hashlib
from pathlib import Path
import unittest

import pandas as pd

from conceptbasis.splits import image_class


ROOT = Path(__file__).resolve().parents[1]
class RepositoryTests(unittest.TestCase):
    def test_dictionary_is_well_formed(self):
        artifacts = (
            ("dictionary.json", "dictionary_provenance.json"),
            (
                "dictionary_positive_only_edge_average_085.json",
                "dictionary_positive_only_edge_average_085.provenance.json",
            ),
            (
                "dictionary_usage_profile_v8.json",
                "dictionary_usage_profile_v8.provenance.json",
            ),
        )
        for dictionary_name, provenance_name in artifacts:
            with self.subTest(dictionary=dictionary_name):
                path = ROOT / "data" / dictionary_name
                dictionary = json.loads(path.read_text())
                self.assertEqual(len(dictionary), 256)
                self.assertEqual(len({concept["name"] for concept in dictionary}), 256)
                self.assertTrue(all(concept["members"] for concept in dictionary))
                provenance = json.loads((ROOT / "data" / provenance_name).read_text())
                self.assertEqual(
                    provenance["dictionary_sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                self.assertEqual(provenance["attribute_split"], "train")

    def test_reproduction_entrypoints_exist(self):
        expected = [
            "scripts/data/make_class_splits.py",
            "scripts/data/partition_open_tags.py",
            "scripts/data/mine_open_tags.py",
            "scripts/data/caption_images.py",
            "scripts/data/build_training_inputs.py",
            "scripts/data/label_fixed_dictionary.py",
            "scripts/vllm/label_fixed_dictionary.sh",
            "scripts/evaluation/build_retrieval_profiles.py",
            "scripts/evaluation/evaluate_compositional_retrieval.py",
            "scripts/dictionary/build_dictionary.py",
            "scripts/dictionary/propose_merge_edges.py",
            "scripts/visualization/make_trained_playground.py",
            "scripts/visualization/make_fixed_label_matrix.py",
        ]
        for relative in expected:
            with self.subTest(relative=relative):
                self.assertTrue((ROOT / relative).is_file())

    def test_public_playground_coefficients_are_nonnegative(self):
        template = (
            ROOT / "scripts" / "visualization" / "make_trained_playground.py"
        ).read_text()
        self.assertIn("inp.min=0;inp.max=3", template)
        self.assertIn("Math.max(0,Math.min(3,v))", template)

    def test_selected_outputs_share_the_usage_profile_v8_stack(self):
        selected = {
            "conceptbasis/train.py",
            "scripts/data/build_training_inputs.py",
            "scripts/visualization/make_trained_playground.py",
            "scripts/visualization/make_fixed_label_matrix.py",
            "scripts/visualization/make_dictionary_gallery.py",
            "scripts/visualization/make_readme_composability_chart.py",
            "reproduce.sh",
        }
        for relative in selected:
            with self.subTest(relative=relative):
                text = (ROOT / relative).read_text()
                self.assertIn("usage_profile_v8", text)

        open_tags = (
            ROOT / "scripts/visualization/make_open_tag_gallery.py"
        ).read_text()
        self.assertIn("attributes_dev_vllm_gemma4_nvfp4.jsonl", open_tags)
        captions = (ROOT / "scripts/data/caption_images.py").read_text()
        self.assertIn("captions_vllm_gemma4_nvfp4_clip_grounded_v2.jsonl", captions)

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
