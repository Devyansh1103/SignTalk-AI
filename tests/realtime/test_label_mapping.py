"""
Unit tests for class label mapping (Phase 4 Part 2).
Validates consistency between training vocabulary, offline label mapping,
and real-time inference label maps.
"""

import pytest
import os
import json
import csv

from src.data.label_map import (
    LABEL_MAP,
    REVERSE_LABEL_MAP,
    GLOSS_MAP,
    TRANSLATION_MAP,
    NUM_CLASSES,
    get_label,
    get_class_id,
    validate_label_map
)


class TestLabelMapping:
    def test_internal_mapping_consistency(self):
        assert len(LABEL_MAP) == NUM_CLASSES
        assert len(REVERSE_LABEL_MAP) == NUM_CLASSES
        assert len(GLOSS_MAP) == NUM_CLASSES
        assert len(TRANSLATION_MAP) == NUM_CLASSES

        # Bidirectional mapping integrity
        for cid, lbl in LABEL_MAP.items():
            assert REVERSE_LABEL_MAP[lbl] == cid
            assert get_label(cid) == lbl
            assert get_class_id(lbl) == cid

    def test_consistency_with_mvp_10_json(self):
        vocab_path = "assets/vocabularies/mvp_10.json"
        if not os.path.exists(vocab_path):
            pytest.skip("MVP 10 vocabulary file not found")

        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_data = json.load(f)

        glosses = vocab_data.get("class_id_to_gloss", {})
        translations = vocab_data.get("class_id_to_translation", {})
        assert len(glosses) == NUM_CLASSES

        for str_cid, gloss in glosses.items():
            cid = int(str_cid)
            assert GLOSS_MAP[cid] == gloss
            assert TRANSLATION_MAP[cid] == translations[str_cid]

    def test_consistency_with_label_mapping_csv(self):
        csv_path = "data/processed/label_mapping.csv"
        if not os.path.exists(csv_path):
            pytest.skip("Processed label_mapping.csv not found")

        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == NUM_CLASSES
        for row in rows:
            cid = int(row["canonical_class_id"])
            lbl = row["canonical_label"]
            assert LABEL_MAP[cid] == lbl

    def test_validate_label_map_helper(self):
        assert validate_label_map() is True
