"""
parse_wit3.py -- parse the WIT3/IWSLT TED English (2016 release) XML into a
list of talk records, tolerating malformed XML.

Each <file> element becomes a dict:
    talkid, speaker, date, title, keywords (list[str]), wordnum, charnum,
    content (str, the full subtitle text)

Usage:
    python3 scripts/parse_wit3.py data/extracted/ted_en-20160408.xml data/processed/talks.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import xml.etree.ElementTree as ET


def _text(elem, tag, default=None):
    child = elem.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _int(elem, tag, default=None):
    v = _text(elem, tag)
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_file_elem(file_elem) -> dict | None:
    # NOTE: <content> (the full subtitle text) is a SIBLING of <head> within
    # <file>, not nested inside <head> -- verified directly against the 2016
    # release XML (the brief's description of "a <content> field" inside
    # each <file> was right, just not nested under <head> as first assumed).
    head = file_elem.find("head")
    if head is None:
        return None
    talkid = _int(head, "talkid")
    speaker = _text(head, "speaker")
    date = _text(head, "date")
    title = _text(head, "title")
    keywords_raw = _text(head, "keywords", "")
    keywords = [k.strip() for k in keywords_raw.split(",") if k.strip()]
    wordnum = _int(head, "wordnum")
    charnum = _int(head, "charnum")
    content_elem = file_elem.find("content")
    content = content_elem.text if (content_elem is not None and content_elem.text) else None
    if content is None or talkid is None:
        return None
    return {
        "talkid": talkid,
        "speaker": speaker,
        "date": date,
        "title": title,
        "keywords": keywords,
        "wordnum": wordnum,
        "charnum": charnum,
        "content": content.strip(),
    }


def parse_wit3_xml(path: str) -> list[dict]:
    """Parse the whole file; if that fails (malformed XML), fall back to
    per-<file>-block recovery via regex splitting so a single bad talk
    doesn't sink the whole corpus."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        records = []
        for file_elem in root.findall("file"):
            rec = parse_file_elem(file_elem)
            if rec is not None:
                records.append(rec)
        return records
    except ET.ParseError as e:
        print(f"WARNING: whole-file parse failed ({e}); falling back to per-file recovery", file=sys.stderr)
        with open(path, encoding="utf-8", errors="replace") as f:
            raw = f.read()
        blocks = re.findall(r"<file id=\"\d+\">.*?</file>", raw, re.S)
        records = []
        skipped = 0
        for block in blocks:
            try:
                elem = ET.fromstring("<file>" + block.split(">", 1)[1])
            except ET.ParseError:
                skipped += 1
                continue
            rec = parse_file_elem(elem)
            if rec is not None:
                records.append(rec)
            else:
                skipped += 1
        print(f"Recovered {len(records)} talks, skipped {skipped} malformed blocks", file=sys.stderr)
        return records


if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "data/extracted/ted_en-20160408.xml"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "data/processed/talks.jsonl"
    records = parse_wit3_xml(in_path)
    print(f"Parsed {len(records)} talks from {in_path}")
    import os

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Wrote {out_path}")
