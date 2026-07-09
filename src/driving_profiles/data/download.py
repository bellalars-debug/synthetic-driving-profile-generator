"""Fetch and verify the NHTS 2022 public-use data extract.

Per docs/data_requirements.md section 5: fetch the CSV bundle, verify it
contains the four core travel files this project's pipeline depends on,
and persist a sha256 + source URL + Last-Modified alongside the extract so
it's reproducible from a fresh clone.

## Filename check: a deliberate change from "count only"

This module originally verified only that the archive held exactly four
CSVs, without checking their names, since the official docs and
third-party sources once disagreed on whether they were `hhpub.csv`-style
or `hhv2pub.csv`-style. `ingest.py` has since settled that ambiguity - it
already hard-codes the `*v2pub.csv` names - so a pure count check is no
longer the right guard: the official archive now ships a fifth CSV,
`ldtv2pub.csv` (long-distance trips), which `clean.py`/`ingest.py`
intentionally never load. A count of 4 would reject today's real archive;
a count of 5 would silently accept a future archive that dropped or
renamed one of the four files this pipeline actually reads. So the check
is now name-based: `REQUIRED_CSV_FILENAMES` must all be present (or
`fetch` fails), `KNOWN_OPTIONAL_CSV_FILENAMES` may or may not be present,
and anything else is treated as an unrecognized archive-layout change and
rejected rather than silently trusted.
"""

from __future__ import annotations

import hashlib
import json
import logging
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen

logger = logging.getLogger(__name__)

NHTS_2022_CSV_URL = "https://nhts.ornl.gov/assets/2022/download/csv.zip"

# The four travel files ingest.py reads (household, person, vehicle, trip -
# see its own HOUSEHOLD_ID_COLUMNS etc.). fetch() fails if any is missing.
REQUIRED_CSV_FILENAMES = frozenset(
    {"hhv2pub.csv", "perv2pub.csv", "vehv2pub.csv", "tripv2pub.csv"}
)

# Official files the NHTS 2022 archive may also include that this
# project's pipeline intentionally never loads (see clean.py's module
# docstring for ldtv2pub.csv specifically). Present or absent, they don't
# affect whether fetch() succeeds - they're just not treated as a sign of
# an unrecognized archive layout the way a wholly unexpected file is.
KNOWN_OPTIONAL_CSV_FILENAMES = frozenset({"ldtv2pub.csv"})

DEFAULT_DEST_DIR = Path("data/raw")
MANIFEST_FILENAME = "manifest.json"


@dataclass
class DownloadManifest:
    source_url: str
    sha256: str
    last_modified: str | None
    downloaded_at: str
    extracted_files: list[str]


def _download(url: str) -> tuple[bytes, str | None]:
    """Fetch `url` and return its raw bytes plus the Last-Modified header, if any."""
    with urlopen(url) as response:  # noqa: S310 - fixed, hard-coded HTTPS URL
        content = response.read()
        last_modified = response.headers.get("Last-Modified")
    return content, last_modified


def _extract_csvs(zip_bytes: bytes, dest_dir: Path) -> list[str]:
    """Extract every CSV member of `zip_bytes` (flattened) into `dest_dir`.

    Raises ValueError if any of `REQUIRED_CSV_FILENAMES` is missing, or if
    the archive contains a CSV that is neither required nor a recognized
    optional file (`KNOWN_OPTIONAL_CSV_FILENAMES`) - either case means an
    assumption in docs/data_requirements.md no longer holds and needs
    re-checking before ingest.py trusts the contents of dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(BytesIO(zip_bytes)) as archive:
        csv_members = [
            member
            for member in archive.infolist()
            if not member.is_dir() and member.filename.lower().endswith(".csv")
        ]
        for member in csv_members:
            name = Path(member.filename).name
            with archive.open(member) as source, open(dest_dir / name, "wb") as target:
                target.write(source.read())
            extracted.append(name)

    logger.info("Extracted %d CSV file(s): %s", len(extracted), sorted(extracted))

    extracted_set = set(extracted)
    missing = REQUIRED_CSV_FILENAMES - extracted_set
    unexpected = extracted_set - REQUIRED_CSV_FILENAMES - KNOWN_OPTIONAL_CSV_FILENAMES
    if missing or unexpected:
        raise ValueError(
            "The NHTS 2022 zip's contents don't match this project's expectations "
            f"(found {sorted(extracted_set)}): "
            f"missing required file(s) {sorted(missing)}; "
            f"unrecognized file(s) {sorted(unexpected)}. "
            "The archive layout may have changed since docs/data_requirements.md "
            "was written - re-verify before trusting these files."
        )
    return extracted


def fetch(
    dest_dir: Path = DEFAULT_DEST_DIR, url: str = NHTS_2022_CSV_URL
) -> DownloadManifest:
    """Download, extract, and verify the NHTS 2022 public-use CSV bundle.

    Writes the extracted CSVs and a manifest.json (source URL, sha256,
    Last-Modified, extracted filenames) into `dest_dir`.
    """
    logger.info("Downloading NHTS 2022 data from %s", url)
    content, last_modified = _download(url)
    sha256 = hashlib.sha256(content).hexdigest()

    extracted_files = _extract_csvs(content, dest_dir)

    manifest = DownloadManifest(
        source_url=url,
        sha256=sha256,
        last_modified=last_modified,
        downloaded_at=datetime.now(UTC).isoformat(),
        extracted_files=sorted(extracted_files),
    )
    manifest_path = dest_dir / MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(asdict(manifest), indent=2) + "\n")
    logger.info("Wrote manifest to %s", manifest_path)

    return manifest


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fetch()
