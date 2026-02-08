#!/usr/bin/env python3
import os
import shutil
import csv
from pathlib import Path

"""
Copy SOCOFing Real and Altered images for subjects 1..500 into project
fingerprint organized structure and produce a mapping CSV.

Usage (defaults are already set for your environment):
    python scripts/import_socofing.py
"""

SOURCE_ROOT = r"D:\DownLoad\archive\SOCOFing"
TARGET_ROOT = r"d:\VSpro\1.0\.idea\src\data\fingerprint\fingerprint\organized"
SUBJECT_START = 1
SUBJECT_END = 500

ALTERED_DIRS = {
    "Easy": os.path.join(SOURCE_ROOT, "Altered", "Altered-Easy"),
    "Medium": os.path.join(SOURCE_ROOT, "Altered", "Altered-Medium"),
    "Hard": os.path.join(SOURCE_ROOT, "Altered", "Altered-Hard"),
}
REAL_DIR = os.path.join(SOURCE_ROOT, "Real")

KNOWN_ALTERATIONS = {"CR", "Obl", "Zcut"}
FINGER_KEYS = {"index", "little", "middle", "ring", "thumb"}

mapping_rows = []
copied_count = 0
skipped_count = 0

def ensure_dirs(subject_id_padded):
    left = os.path.join(TARGET_ROOT, subject_id_padded, "left")
    right = os.path.join(TARGET_ROOT, subject_id_padded, "right")
    os.makedirs(left, exist_ok=True)
    os.makedirs(right, exist_ok=True)
    return left, right

def parse_filename(fname):
    # Examples:
    # 1__M_Left_index_finger.BMP
    # 1__M_Left_index_finger_CR.BMP
    base = os.path.splitext(fname)[0]
    if "__" not in base:
        return None
    subject, rest = base.split("__", 1)
    tokens = rest.split("_")
    if len(tokens) < 3:
        return None
    sex = tokens[0]
    hand = tokens[1].lower()
    # finger may be two tokens like index + finger
    alteration = ""
    if tokens[-1] in KNOWN_ALTERATIONS:
        alteration = tokens[-1]
        finger_tokens = tokens[2:-1]
    else:
        finger_tokens = tokens[2:]
    finger = "_".join(finger_tokens)
    return {
        "subject": subject,
        "sex": sex,
        "hand": hand,
        "finger": finger,
        "alteration": alteration
    }

def copy_and_record(src_path, subject_padded, parsed, source_label, alteration_level):
    global copied_count, skipped_count
    hand = parsed["hand"]
    finger = parsed["finger"]
    # preserve alteration suffix in filename if present
    alteration_suffix = ("_" + parsed["alteration"]) if parsed["alteration"] else ""
    target_fname = f"{finger}{alteration_suffix}.BMP"
    target_dir = os.path.join(TARGET_ROOT, subject_padded, hand)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, target_fname)
    try:
        shutil.copy2(src_path, target_path)
        copied_count += 1
        mapping_rows.append({
            "filepath": os.path.relpath(target_path, start=TARGET_ROOT),
            "subject_id": subject_padded,
            "hand": hand,
            "finger": finger,
            "source": source_label,
            "alteration_level": alteration_level,
            "original_path": src_path
        })
    except Exception:
        skipped_count += 1

def process_real_for_subject(subject):
    subject_prefix = f"{subject}__"
    # find files in REAL_DIR starting with subject_prefix
    for fname in os.listdir(REAL_DIR):
        if not fname.lower().endswith(".bmp"):
            continue
        if not fname.startswith(subject_prefix):
            continue
        parsed = parse_filename(fname)
        if not parsed:
            continue
        subject_padded = f"{int(parsed['subject']):03d}"
        ensure_dirs(subject_padded)
        copy_and_record(os.path.join(REAL_DIR, fname), subject_padded, parsed, "Real", "")

def process_altered_for_subject(subject):
    subject_prefix = f"{subject}__"
    # each altered level folder contains files with same subject naming
    for level_name, level_dir in ALTERED_DIRS.items():
        if not os.path.isdir(level_dir):
            continue
        for fname in os.listdir(level_dir):
            if not fname.lower().endswith(".bmp"):
                continue
            if not fname.startswith(subject_prefix):
                continue
            parsed = parse_filename(fname)
            if not parsed:
                continue
            subject_padded = f"{int(parsed['subject']):03d}"
            ensure_dirs(subject_padded)
            copy_and_record(os.path.join(level_dir, fname), subject_padded, parsed, "Altered", level_name)

def main():
    print("Starting copy for subjects %d..%d" % (SUBJECT_START, SUBJECT_END))
    for subject in range(SUBJECT_START, SUBJECT_END + 1):
        # process Real
        process_real_for_subject(subject)
        # process Altered Easy/Medium/Hard
        process_altered_for_subject(subject)

    mapping_file = os.path.join("data", "fingerprint", "mapping_socofing_first500.csv")
    mapping_file_full = os.path.join(os.path.dirname(TARGET_ROOT), os.path.basename(mapping_file))
    # ensure parent dir exists
    os.makedirs(os.path.dirname(mapping_file_full), exist_ok=True)
    with open(mapping_file_full, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filepath", "subject_id", "hand", "finger", "source", "alteration_level", "original_path"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in mapping_rows:
            writer.writerow(row)

    print(f"Copied {copied_count} files, skipped {skipped_count}. Mapping written to {mapping_file_full}")

if __name__ == "__main__":
    main()
