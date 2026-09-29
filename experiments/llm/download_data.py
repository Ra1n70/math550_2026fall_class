"""Checksum-aware downloads for the book, SFT, and preference corpora."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import random
import re
import secrets
import time
from urllib.request import Request, urlopen
import zipfile

from math550.topics.llm import strip_gutenberg_boilerplate


USER_AGENT = "MATH550-LLM-course/1.0 (educational reproducibility exercise)"
DEFAULT_BOOK_LIMIT = 16
MAX_CONSECUTIVE_FAILURES = 25
DEFAULT_REQUEST_DELAY = 2.0


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str, *, timeout: int = 120) -> tuple[bytes, str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read(), response.geturl()


def _write_verified(url: str, path: Path, expected_sha256: str | None) -> dict[str, object]:
    if path.is_file():
        data = path.read_bytes()
        observed = _sha256(data)
        if expected_sha256 is None or observed == expected_sha256:
            return {
                "path": str(path),
                "bytes": len(data),
                "sha256": observed,
                "status": "existing",
            }
    data, resolved = _fetch(url)
    observed = _sha256(data)
    if expected_sha256 is not None and observed != expected_sha256:
        raise RuntimeError(
            f"checksum mismatch for {url}: expected {expected_sha256}, observed {observed}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return {
        "path": str(path),
        "requested_url": url,
        "resolved_url": resolved,
        "bytes": len(data),
        "sha256": observed,
        "status": "downloaded",
    }


def _normalized_title(value: str) -> str:
    """Return a conservative key used only to avoid duplicate editions."""

    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _catalog_has_marker(searchable: str, marker: str) -> bool:
    """Match catalog labels without treating ``nonfiction`` as ``fiction``."""

    if marker == "fiction":
        return re.search(r"(?<![a-z])fiction(?![a-z])", searchable) is not None
    return marker in searchable


def _selection_config(section: dict[str, object]) -> dict[str, object]:
    """Return the current selection config, accepting old manifests during upgrades."""

    return dict(
        section.get("catalog_selection", section.get("catalog_expansion", {}))
    )


def _requested_book_count(section: dict[str, object], limit: int | None) -> int:
    config = _selection_config(section)
    requested = (
        int(config.get("default_book_limit", DEFAULT_BOOK_LIMIT))
        if limit is None
        else limit
    )
    if requested < 2:
        raise ValueError(
            "--book-limit must be at least 2 so training and validation use different books"
        )
    return requested


def catalog_training_candidates(
    catalog_gzip: bytes,
    section: dict[str, object],
    excluded_books: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    """Return the eligible population from which a run samples random books."""

    config = _selection_config(section)
    excluded_books = excluded_books or []
    excluded_ids = {int(book["id"]) for book in excluded_books}
    excluded_ids.add(int(section["homework_book"]["id"]))
    excluded_titles = {
        _normalized_title(str(book["title"])) for book in excluded_books
    }
    excluded_titles.add(_normalized_title(str(section["homework_book"]["title"])))
    markers = tuple(
        str(marker).casefold() for marker in config.get("subject_markers", [])
    )
    candidates: list[dict[str, object]] = []
    seen_titles = set(excluded_titles)
    with gzip.GzipFile(fileobj=io.BytesIO(catalog_gzip), mode="rb") as compressed:
        with io.TextIOWrapper(
            compressed, encoding="utf-8-sig", newline=""
        ) as text_stream:
            for row in csv.DictReader(text_stream):
                if row.get("Type") != "Text" or row.get("Language") != "en":
                    continue
                try:
                    book_id = int(row["Text#"])
                except (KeyError, TypeError, ValueError):
                    continue
                title = " ".join((row.get("Title") or "").split())
                author = " ".join((row.get("Authors") or "Unknown").split())
                searchable = (
                    f"{row.get('Subjects', '')}; {row.get('Bookshelves', '')}"
                ).casefold()
                title_key = _normalized_title(title)
                if (
                    not title
                    or book_id in excluded_ids
                    or title_key in seen_titles
                    or (
                        markers
                        and not any(
                            _catalog_has_marker(searchable, marker)
                            for marker in markers
                        )
                    )
                ):
                    continue
                seen_titles.add(title_key)
                candidates.append(
                    {
                        "id": book_id,
                        "title": title,
                        "author": author,
                        "selection_source": "project_gutenberg_weekly_catalog",
                    }
                )
    candidates.sort(key=lambda book: int(book["id"]))
    return candidates


def _catalog_bytes(
    data_root: Path,
    section: dict[str, object],
    *,
    refresh: bool,
) -> tuple[bytes, dict[str, object]]:
    config = _selection_config(section)
    catalog_path = data_root / "raw" / "catalog" / "pg_catalog.csv.gz"
    url = str(config["csv_gzip_url"])
    if catalog_path.is_file() and not refresh:
        data = catalog_path.read_bytes()
        return data, {
            "path": str(catalog_path),
            "requested_url": url,
            "resolved_url": url,
            "bytes": len(data),
            "sha256": _sha256(data),
            "status": "existing",
        }
    data, resolved = _fetch(url)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_bytes(data)
    return data, {
        "path": str(catalog_path),
        "requested_url": url,
        "resolved_url": resolved,
        "bytes": len(data),
        "sha256": _sha256(data),
        "status": "downloaded",
    }


def _book_id_path(book_id: int) -> str:
    digits = str(book_id)
    return "/".join(digits[:-1]) + "/" if len(digits) > 1 else ""


def _archive_templates(
    section: dict[str, object], override: str | None
) -> list[dict[str, str]]:
    if override:
        return [{"url": override, "encoding": "utf-8"}]
    templates = section["url_templates"].get("text_archives", [])
    if not templates:
        raise ValueError("sources.json does not define a text archive download source")
    return [dict(template) for template in templates]


def _extract_book_text(
    payload: bytes, *, book_id: int, preferred_encoding: str
) -> tuple[str, str | None]:
    """Decode a mirror payload, extracting the largest text member from ZIP files."""

    member_name: str | None = None
    content = payload
    if payload.startswith(b"PK"):
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = [
                member
                for member in archive.infolist()
                if not member.is_dir() and member.filename.casefold().endswith(".txt")
            ]
            if not members:
                raise RuntimeError(f"PG{book_id} archive contains no .txt file")
            member = max(members, key=lambda item: item.file_size)
            member_name = member.filename
            content = archive.read(member)
    encodings = [preferred_encoding, "utf-8-sig", "utf-8", "latin-1"]
    for encoding in dict.fromkeys(encodings):
        try:
            return content.decode(encoding), member_name
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"PG{book_id} text could not be decoded")


def _fetch_book_text(
    section: dict[str, object], book_id: int, *, url_template: str | None
) -> tuple[bytes, str, str, str, str | None]:
    errors: list[str] = []
    for template in _archive_templates(section, url_template):
        url = template["url"].format(
            id=book_id, id_path=_book_id_path(book_id)
        )
        try:
            raw, resolved = _fetch(url)
            text, member_name = _extract_book_text(
                raw,
                book_id=book_id,
                preferred_encoding=template.get("encoding", "utf-8"),
            )
            return raw, resolved, url, text, member_name
        except Exception as error:
            errors.append(f"{url}: {error}")
    raise RuntimeError("; ".join(errors))


def _download_book(
    data_root: Path,
    section: dict[str, object],
    book: dict[str, object],
    *,
    previous: dict[str, object] | None,
    request_delay: float,
    book_url_template: str | None = None,
) -> dict[str, object]:
    destination = data_root / "raw" / "books" / f"pg{book['id']}.txt"
    if destination.is_file() and destination.stat().st_size:
        cleaned_sha = _sha256(destination.read_bytes())
        record = {
            **(previous or {}),
            **book,
            "catalog_page": section["url_templates"]["catalog_page"].format(
                id=book["id"]
            ),
            "path": str(destination),
            "cleaned_utf8_bytes": destination.stat().st_size,
            "cleaned_sha256": cleaned_sha,
            "status": "existing",
            "network_request": False,
        }
        if previous and previous.get("cleaned_sha256") not in {None, cleaned_sha}:
            record["previous_cleaned_sha256"] = previous["cleaned_sha256"]
            record["local_file_changed_since_manifest"] = True
        return record
    raw, resolved, url, decoded, archive_member = _fetch_book_text(
        section, int(book["id"]), url_template=book_url_template
    )
    cleaned = strip_gutenberg_boilerplate(decoded)
    if not cleaned.strip():
        raise RuntimeError(f"PG{book['id']} produced an empty cleaned text")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cleaned, encoding="utf-8")
    if request_delay:
        time.sleep(request_delay)
    return {
        **book,
        "catalog_page": section["url_templates"]["catalog_page"].format(id=book["id"]),
        "requested_url": url,
        "resolved_url": resolved,
        "archive_member": archive_member,
        "path": str(destination),
        "download_bytes": len(raw),
        "cleaned_utf8_bytes": destination.stat().st_size,
        "download_sha256": _sha256(raw),
        "cleaned_sha256": _sha256(destination.read_bytes()),
        "status": "downloaded",
        "network_request": True,
    }


def _cached_candidate_ids(data_root: Path, eligible_ids: set[int]) -> set[int]:
    cached: set[int] = set()
    books_dir = data_root / "raw" / "books"
    if not books_dir.is_dir():
        return cached
    for path in books_dir.glob("pg*.txt"):
        match = re.fullmatch(r"pg([0-9]+)\.txt", path.name)
        if match and path.stat().st_size:
            book_id = int(match.group(1))
            if book_id in eligible_ids:
                cached.add(book_id)
    return cached


def _validation_book_count(total: int, fraction: float) -> int:
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    return min(total - 1, max(1, int(total * fraction + 0.5)))


def download_books(
    data_root: Path,
    sources: dict[str, object],
    limit: int | None = None,
    *,
    previous_books: list[dict[str, object]] | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    refresh_catalog: bool = False,
    selection_seed: int | None = None,
    book_url_template: str | None = None,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Randomly select books, preferring cached files, and return the run record."""

    section = sources["project_gutenberg"]
    if request_delay < 0:
        raise ValueError("request_delay must be non-negative")
    if max_consecutive_failures < 1:
        raise ValueError("max_consecutive_failures must be positive")
    requested = _requested_book_count(section, limit)
    config = _selection_config(section)
    actual_seed = secrets.randbits(64) if selection_seed is None else selection_seed
    rng = random.Random(actual_seed)
    catalog_data, catalog_record = _catalog_bytes(
        data_root, section, refresh=refresh_catalog
    )
    candidates = catalog_training_candidates(catalog_data, section)
    if requested > len(candidates):
        raise ValueError(
            f"--book-limit {requested} exceeds the {len(candidates)} eligible books "
            "in the cached Project Gutenberg catalog"
        )
    candidate_by_id = {int(book["id"]): book for book in candidates}
    cached_ids = _cached_candidate_ids(data_root, set(candidate_by_id))
    cached = [candidate_by_id[book_id] for book_id in sorted(cached_ids)]
    uncached = [book for book in candidates if int(book["id"]) not in cached_ids]
    rng.shuffle(cached)
    rng.shuffle(uncached)
    # Cache-first random order means increasing N downloads only the missing
    # difference while a repeated/smaller run makes no book network requests.
    candidate_order = cached + uncached
    previous_by_id = {
        int(book["id"]): book for book in (previous_books or []) if "id" in book
    }
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    consecutive_failures = 0
    for book in candidate_order:
        if len(records) >= requested:
            break
        try:
            record = _download_book(
                data_root,
                section,
                book,
                previous=previous_by_id.get(int(book["id"])),
                request_delay=request_delay,
                book_url_template=book_url_template,
            )
        except Exception as error:
            failures.append(
                {"id": book["id"], "title": book["title"], "error": str(error)}
            )
            consecutive_failures += 1
            print(f"[skip] PG{book['id']}: {error}")
            if consecutive_failures >= max_consecutive_failures:
                print(
                    f"[stop] reached {max_consecutive_failures} consecutive failures"
                )
                break
            continue
        consecutive_failures = 0
        records.append(record)
        action = "reuse" if record.get("status") == "existing" else "download"
        print(
            f"[{len(records):0{len(str(requested))}d}/{requested}] "
            f"[{action}] PG{book['id']}: {book['title']}"
        )
    if len(records) != requested:
        raise RuntimeError(
            f"requested {requested} books but downloaded {len(records)}; "
            f"catalog candidates failed={len(failures)}"
        )
    validation_count = _validation_book_count(
        requested, float(config.get("validation_fraction", 0.1))
    )
    validation_ids = {
        int(book["id"]) for book in rng.sample(records, validation_count)
    }
    for book in records:
        book["split"] = (
            "validation" if int(book["id"]) in validation_ids else "train"
        )
    selection = {
        "requested_book_limit": requested,
        "actual_books": len(records),
        "training_books": sum(book["split"] == "train" for book in records),
        "validation_books": sum(book["split"] == "validation" for book in records),
        "validation_policy": (
            "random whole-book split with at least one book per split; "
            f"target validation fraction {config.get('validation_fraction', 0.1)}"
        ),
        "selection_policy": "random cache-first sample without replacement",
        "selection_seed": actual_seed,
        "eligible_catalog_books": len(candidates),
        "eligible_cached_books": len(cached),
        "existing_books_reused": sum(
            book.get("status") == "existing" for book in records
        ),
        "new_books_downloaded": sum(
            book.get("status") == "downloaded" for book in records
        ),
        "catalog": catalog_record,
        "catalog_selection": config,
        "failed_candidates": failures,
        "request_delay_seconds": request_delay,
        "consecutive_failure_cap": max_consecutive_failures,
        "book_url_template_override": book_url_template,
    }
    return records, selection


def download_homework_book(
    data_root: Path,
    sources: dict[str, object],
    *,
    book_url_template: str | None = None,
) -> dict[str, object]:
    section = sources["project_gutenberg"]
    book = section["homework_book"]
    destination = data_root / "raw" / "homework" / f"pg{book['id']}.txt"
    if destination.is_file() and destination.stat().st_size:
        data = destination.read_bytes()
        return {
            **book,
            "catalog_page": section["url_templates"]["catalog_page"].format(
                id=book["id"]
            ),
            "path": str(destination),
            "cleaned_utf8_bytes": len(data),
            "cleaned_sha256": _sha256(data),
            "status": "existing",
            "network_request": False,
        }
    raw, resolved, url, decoded, archive_member = _fetch_book_text(
        section, int(book["id"]), url_template=book_url_template
    )
    cleaned = strip_gutenberg_boilerplate(decoded)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cleaned, encoding="utf-8")
    return {
        **book,
        "catalog_page": section["url_templates"]["catalog_page"].format(id=book["id"]),
        "requested_url": url,
        "resolved_url": resolved,
        "archive_member": archive_member,
        "path": str(destination),
        "download_bytes": len(raw),
        "cleaned_utf8_bytes": destination.stat().st_size,
        "download_sha256": _sha256(raw),
        "cleaned_sha256": _sha256(destination.read_bytes()),
        "status": "downloaded",
        "network_request": True,
    }


def run(
    dataset: str,
    data_root: Path,
    book_limit: int | None = None,
    request_delay: float = DEFAULT_REQUEST_DELAY,
    refresh_catalog: bool = False,
    selection_seed: int | None = None,
    book_url_template: str | None = None,
    max_consecutive_failures: int = MAX_CONSECUTIVE_FAILURES,
) -> Path:
    sources = json.loads((data_root / "sources.json").read_text(encoding="utf-8"))
    manifest_path = data_root / "download_manifest.json"
    records: dict[str, object] = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"sources_file": str(data_root / "sources.json")}
    )
    if dataset in {"books", "all"}:
        books, selection = download_books(
            data_root,
            sources,
            book_limit,
            previous_books=records.get("books", []),
            request_delay=request_delay,
            refresh_catalog=refresh_catalog,
            selection_seed=selection_seed,
            book_url_template=book_url_template,
            max_consecutive_failures=max_consecutive_failures,
        )
        records["books"] = books
        records["book_selection"] = selection
    if dataset in {"sft", "all"}:
        source = sources["sft"]
        records["sft"] = _write_verified(
            source["download_url"],
            data_root / "raw" / "databricks-dolly-15k.jsonl",
            source["sha256"],
        )
    if dataset in {"preference", "all"}:
        source = sources["preference"]
        records["preference"] = _write_verified(
            source["download_url"],
            data_root / "raw" / "hh-rlhf-harmless-base-train.jsonl.gz",
            source["sha256"],
        )
    if dataset in {"homework", "all"}:
        records["homework"] = download_homework_book(
            data_root, sources, book_url_template=book_url_template
        )
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("books", "sft", "preference", "homework", "all"),
        default="books",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/llm"))
    parser.add_argument(
        "--book-limit",
        type=int,
        help=(
            "total randomly selected books (minimum 2; default 16; no artificial maximum)"
        ),
    )
    parser.add_argument(
        "--selection-seed",
        type=int,
        help="optional reproducible random seed; an unpredictable seed is recorded by default",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=DEFAULT_REQUEST_DELAY,
        help="seconds between new mirror downloads (default: 2)",
    )
    parser.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="replace the cached official weekly catalog before selecting expansion books",
    )
    parser.add_argument(
        "--book-url-template",
        help=(
            "optional authorized mirror URL template with {id} and optional {id_path}; "
            "ZIP or plain-text payloads are accepted"
        ),
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=MAX_CONSECUTIVE_FAILURES,
        help=(
            "abort after this many consecutive book failures "
            f"(default: {MAX_CONSECUTIVE_FAILURES})"
        ),
    )
    args = parser.parse_args()
    if args.request_delay < 0:
        parser.error("--request-delay must be non-negative")
    print(
        run(
            args.dataset,
            args.data_root,
            args.book_limit,
            request_delay=args.request_delay,
            refresh_catalog=args.refresh_catalog,
            selection_seed=args.selection_seed,
            book_url_template=args.book_url_template,
            max_consecutive_failures=args.max_consecutive_failures,
        )
    )


if __name__ == "__main__":
    main()
