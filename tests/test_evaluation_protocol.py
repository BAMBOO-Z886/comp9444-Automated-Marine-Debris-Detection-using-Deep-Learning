import copy
import unittest
import warnings

from help_funciton.evaluation_protocol import audit_splits


def split(image_id, video, annotated=True):
    return {
        "images": [{"id": image_id, "file_name": f"{video}_frame0001.jpg"}],
        "categories": [{"id": 7, "name": "debris"}],
        "annotations": [{"image_id": image_id, "category_id": 7}] if annotated else [],
    }


class SplitProtocolTests(unittest.TestCase):
    def test_disjoint_splits_preserve_negative_images(self):
        result = audit_splits(split(1, "train"), split(2, "test", False), split(1, "val"))
        self.assertEqual(result["splits"]["test"]["empty_images"], 1)
        self.assertEqual(result["warnings"], [])

    def test_validation_video_overlap_is_reported_without_mutation(self):
        validation = split(3, "test")
        validation["images"][0]["file_name"] = "test_frame0002.jpg"
        before = copy.deepcopy(validation)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = audit_splits(split(1, "train"), split(2, "test"), validation)
        self.assertEqual(len(caught), 1)
        self.assertEqual(result["overlap"]["validation_test"],
                         {"filenames": 0, "videos": 1, "validation_images": 1})
        self.assertEqual(validation, before)

    def test_train_test_leakage_is_rejected(self):
        for test in (split(1, "test"), split(2, "train")):
            with self.subTest(test=test), self.assertRaises(ValueError):
                audit_splits(split(1, "train"), test)

    def test_taxonomy_mismatch_is_rejected(self):
        test = split(2, "test")
        test["categories"][0]["name"] = "person"
        with self.assertRaisesRegex(ValueError, "taxonomy"):
            audit_splits(split(1, "train"), test)

    def test_empty_split_is_rejected(self):
        test = split(2, "test")
        test["images"] = []
        with self.assertRaisesRegex(ValueError, "nonempty"):
            audit_splits(split(1, "train"), test)


if __name__ == "__main__":
    unittest.main()
