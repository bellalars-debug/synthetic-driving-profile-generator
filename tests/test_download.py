"""Tests for driving_profiles.data.download."""

import hashlib
import json
import zipfile
from io import BytesIO

import pytest

from driving_profiles.data import download


def _make_zip(files: dict[str, str]) -> bytes:
    """Build in-memory zip bytes from {path_in_zip: content}."""
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


FOUR_CSVS = {
    "csv/hhpub.csv": "HOUSEID\n1\n",
    "csv/perpub.csv": "HOUSEID,PERSONID\n1,01\n",
    "csv/vehpub.csv": "HOUSEID,VEHID\n1,01\n",
    "csv/trippub.csv": "HOUSEID,PERSONID,TRIPID\n1,01,01\n",
}


def test_extract_csvs_flattens_nested_paths_and_ignores_non_csv(tmp_path):
    zip_bytes = _make_zip({**FOUR_CSVS, "csv/readme.txt": "not a csv"})

    extracted = download._extract_csvs(zip_bytes, tmp_path)

    assert sorted(extracted) == ["hhpub.csv", "perpub.csv", "trippub.csv", "vehpub.csv"]
    assert (tmp_path / "hhpub.csv").read_text() == "HOUSEID\n1\n"
    assert not (tmp_path / "readme.txt").exists()


def test_extract_csvs_wrong_count_raises(tmp_path):
    zip_bytes = _make_zip({"csv/hhpub.csv": "HOUSEID\n1\n"})

    with pytest.raises(ValueError, match="Expected 4 CSVs"):
        download._extract_csvs(zip_bytes, tmp_path)


def test_extract_csvs_does_not_hardcode_filenames(tmp_path):
    """The archive may use a v2 naming scheme; extraction shouldn't assume names."""
    renamed = {
        "csv/hhv2pub.csv": FOUR_CSVS["csv/hhpub.csv"],
        "csv/perv2pub.csv": FOUR_CSVS["csv/perpub.csv"],
        "csv/vehpub.csv": FOUR_CSVS["csv/vehpub.csv"],
        "csv/trippub.csv": FOUR_CSVS["csv/trippub.csv"],
    }
    zip_bytes = _make_zip(renamed)

    extracted = download._extract_csvs(zip_bytes, tmp_path)

    assert sorted(extracted) == ["hhv2pub.csv", "perv2pub.csv", "trippub.csv", "vehpub.csv"]


def test_fetch_writes_manifest_with_sha256_and_source_info(tmp_path, monkeypatch):
    zip_bytes = _make_zip(FOUR_CSVS)
    expected_sha256 = hashlib.sha256(zip_bytes).hexdigest()

    monkeypatch.setattr(
        download, "_download", lambda url: (zip_bytes, "Fri, 20 Dec 2024 00:00:00 GMT")
    )

    manifest = download.fetch(dest_dir=tmp_path, url="https://example.invalid/csv.zip")

    assert manifest.source_url == "https://example.invalid/csv.zip"
    assert manifest.sha256 == expected_sha256
    assert manifest.last_modified == "Fri, 20 Dec 2024 00:00:00 GMT"
    assert manifest.extracted_files == ["hhpub.csv", "perpub.csv", "trippub.csv", "vehpub.csv"]

    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    on_disk = json.loads(manifest_path.read_text())
    assert on_disk["sha256"] == expected_sha256
    assert on_disk["extracted_files"] == manifest.extracted_files
