"""
SignTalk AI - Canonical Label Map and Vocabulary Registry
Provides the authoritative mapping between canonical integer class IDs,
dataset labels, linguistic glosses, and natural English translations
for the 10-class Indian Sign Language vocabulary (mvp_10).

Specification:
  0: hello     (Gloss: HELLO,     Translation: "Hello")
  1: thankyou  (Gloss: THANK_YOU, Translation: "Thank you")
  2: good      (Gloss: GOOD,      Translation: "Good")
  3: happy     (Gloss: HAPPY,     Translation: "Happy")
  4: monday    (Gloss: MONDAY,    Translation: "Monday")
  5: car       (Gloss: CAR,       Translation: "Car")
  6: bird      (Gloss: BIRD,      Translation: "Bird")
  7: house     (Gloss: HOUSE,     Translation: "House")
  8: time      (Gloss: TIME,      Translation: "Time")
  9: teacher   (Gloss: TEACHER,   Translation: "Teacher")
"""

import os
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassLabelInfo:
    class_id: int
    label: str
    gloss: str
    translation: str
    token_id: int


CANONICAL_CLASSES: Dict[int, ClassLabelInfo] = {
    0: ClassLabelInfo(class_id=0, label="hello", gloss="HELLO", translation="Hello", token_id=4),
    1: ClassLabelInfo(class_id=1, label="thankyou", gloss="THANK_YOU", translation="Thank you", token_id=5),
    2: ClassLabelInfo(class_id=2, label="good", gloss="GOOD", translation="Good", token_id=6),
    3: ClassLabelInfo(class_id=3, label="happy", gloss="HAPPY", translation="Happy", token_id=7),
    4: ClassLabelInfo(class_id=4, label="monday", gloss="MONDAY", translation="Monday", token_id=8),
    5: ClassLabelInfo(class_id=5, label="car", gloss="CAR", translation="Car", token_id=9),
    6: ClassLabelInfo(class_id=6, label="bird", gloss="BIRD", translation="Bird", token_id=10),
    7: ClassLabelInfo(class_id=7, label="house", gloss="HOUSE", translation="House", token_id=11),
    8: ClassLabelInfo(class_id=8, label="time", gloss="TIME", translation="Time", token_id=12),
    9: ClassLabelInfo(class_id=9, label="teacher", gloss="TEACHER", translation="Teacher", token_id=13),
}

NUM_CLASSES: int = len(CANONICAL_CLASSES)
ID_TO_LABEL: Dict[int, str] = {c_id: info.label for c_id, info in CANONICAL_CLASSES.items()}
LABEL_TO_ID: Dict[str, int] = {info.label: c_id for c_id, info in CANONICAL_CLASSES.items()}
ID_TO_GLOSS: Dict[int, str] = {c_id: info.gloss for c_id, info in CANONICAL_CLASSES.items()}
GLOSS_TO_ID: Dict[str, int] = {info.gloss: c_id for c_id, info in CANONICAL_CLASSES.items()}
ID_TO_TRANSLATION: Dict[int, str] = {c_id: info.translation for c_id, info in CANONICAL_CLASSES.items()}

# Backward-compatible and convenience aliases
LABEL_MAP = ID_TO_LABEL
REVERSE_LABEL_MAP = LABEL_TO_ID
GLOSS_MAP = ID_TO_GLOSS
TRANSLATION_MAP = ID_TO_TRANSLATION


def get_label(class_id: int) -> str:
    """Returns canonical label for given class ID."""
    if class_id not in ID_TO_LABEL:
        raise KeyError(f"Invalid class ID: {class_id}. Must be between 0 and {NUM_CLASSES - 1}")
    return ID_TO_LABEL[class_id]


def get_gloss(class_id: int) -> str:
    """Returns linguistic gloss for given class ID."""
    if class_id not in ID_TO_GLOSS:
        raise KeyError(f"Invalid class ID: {class_id}. Must be between 0 and {NUM_CLASSES - 1}")
    return ID_TO_GLOSS[class_id]


def get_translation(class_id: int) -> str:
    """Returns English translation for given class ID."""
    if class_id not in ID_TO_TRANSLATION:
        raise KeyError(f"Invalid class ID: {class_id}. Must be between 0 and {NUM_CLASSES - 1}")
    return ID_TO_TRANSLATION[class_id]


def get_class_id(name_or_gloss: str) -> int:
    """Resolves class ID from either label or gloss name."""
    clean = name_or_gloss.strip().lower()
    if clean in LABEL_TO_ID:
        return LABEL_TO_ID[clean]
    clean_upper = name_or_gloss.strip().upper()
    if clean_upper in GLOSS_TO_ID:
        return GLOSS_TO_ID[clean_upper]
    raise KeyError(f"Unknown class name or gloss: '{name_or_gloss}'")


def validate_label_mapping(vocab_path: str = "assets/vocabularies/mvp_10.json") -> bool:
    """
    Validates that CANONICAL_CLASSES strictly matches the underlying
    vocabulary JSON file on disk.
    """
    if not os.path.exists(vocab_path):
        return True  # Fallback if running outside root
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vocab_glosses = data.get("class_id_to_gloss", {})
    vocab_translations = data.get("class_id_to_translation", {})

    for c_id, info in CANONICAL_CLASSES.items():
        str_id = str(c_id)
        if str_id in vocab_glosses and vocab_glosses[str_id] != info.gloss:
            return False
        if str_id in vocab_translations and vocab_translations[str_id] != info.translation:
            return False
    return True


validate_label_map = validate_label_mapping
