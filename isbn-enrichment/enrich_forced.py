#!/usr/bin/env python3
"""Aggressive ISBN assignment for every row in the 806-book catalogue.

The catalogue owner explicitly wants one ISBN assigned to every row. The script
therefore prefers an exact/probable physical edition when the metadata supports
one, otherwise selects the latest ISBN-bearing edition of the matched work, and
finally falls back to progressively broader real ISBN-bearing title/author
matches. Every weak assumption is labelled clearly.
"""
from __future__ import annotations

import concurrent.futures
import json
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from enrich import (
    author_similarity,
    canonical_isbns,
    first_year,
    isbn10_to_13,
    isbn13_to_10,
    load_books,
    norm,
    primary_author,
    similarity,
    tokens,
    useful_author,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_forced"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GOOGLE_URL = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
USER_AGENT = "JoshElgarBookCatalogueISBN/2.0 (GitHub Actions; forced latest-edition assignment)"
CONTACT_EMAIL = "joshelgar@gmail.com"

_RATE_LOCKS = {"google": threading.Lock(), "openlibrary": threading.Lock()}
_LAST_REQUEST = {"google": 0.0, "openlibrary": 0.0}
_MIN_INTERVAL = {"google": 0.14, "openlibrary": 0.36}

_CACHE_LOCK = threading.Lock()
_HTTP_CACHE: dict[str, dict[str, Any] | None] = {}


def paced(source: str) -> None:
    with _RATE_LOCKS[source]:
        now = time.monotonic()
        wait = _MIN_INTERVAL[source] - (now - _LAST_REQUEST[source])
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST[source] = time.monotonic()


def http_json(url: str, params: dict[str, Any], source: str) -> dict[str, Any] | None:
    full_url = url + "?" + urllib.parse.urlencode(params, doseq=True)
    with _CACHE_LOCK:
        if full_url in _HTTP_CACHE:
            return _HTTP_CACHE[full_url]

    for attempt in range(5):
        paced(source)
        request = urllib.request.Request(
            full_url,
            headers={
                "User-Agent": USER_AGENT,
                "From": CONTACT_EMAIL,
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
                with _CACHE_LOCK:
                    _HTTP_CACHE[full_url] = payload
                return payload
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) + random.random()
                time.sleep(delay)
                continue
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep((2 ** attempt) + random.random())

    with _CACHE_LOCK:
        _HTTP_CACHE[full_url] = None
    return None


def clean_query_text(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"\([^)]*(uncertain|editor|translator|compiler|first name)[^)]*\)", " ", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def useful_publisher(value: Any) -> bool:
    text = norm(value)
    if not text:
        return False
    return not any(term in text for term in ("unknown", "uncertain", "edition imprint", "edition uncertain"))


def parse_google_item(item: dict[str, Any]) -> dict[str, Any] | None:
    info = item.get("volumeInfo") or {}
    identifiers = [str(entry.get("identifier") or "") for entry in info.get("industryIdentifiers", []) or []]
    isbn13, isbn10, all_codes = canonical_isbns(identifiers)
    if not (isbn13 or isbn10):
        return None
    return {
        "source": "Google Books",
        "source_url": str(info.get("infoLink") or item.get("selfLink") or ""),
        "title": str(info.get("title") or ""),
        "subtitle": str(info.get("subtitle") or ""),
        "authors": [str(value) for value in (info.get("authors") or [])],
        "publisher": str(info.get("publisher") or ""),
        "year": first_year(info.get("publishedDate")),
        "isbn13": isbn13,
        "isbn10": isbn10,
        "all_isbns": all_codes,
    }


def google_search(query: str, newest: bool = True, max_results: int = 40) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "q": query,
        "maxResults": max_results,
        "printType": "books",
        "projection": "full",
    }
    if newest:
        params["orderBy"] = "newest"
    payload = http_json(GOOGLE_URL, params, "google") or {}
    candidates: list[dict[str, Any]] = []
    for item in payload.get("items", []) or []:
        candidate = parse_google_item(item)
        if candidate:
            candidates.append(candidate)
    return candidates


def google_queries(book: dict[str, Any]) -> list[tuple[str, str]]:
    title = clean_query_text(book.get("title"))
    author = clean_query_text(primary_author(book.get("author")))
    publisher = clean_query_text(book.get("publisher")) if useful_publisher(book.get("publisher")) else ""

    queries: list[tuple[str, str]] = []
    if title and author:
        queries.append((f'intitle:"{title}" inauthor:"{author}"', "title+author"))
    if title and publisher:
        queries.append((f'intitle:"{title}" inpublisher:"{publisher}"', "title+publisher"))
    if title:
        queries.append((f'intitle:"{title}"', "title-only"))

    meaningful = [word for word in norm(title).split() if word not in {
        "the", "a", "an", "of", "and", "to", "in", "for", "with", "from", "by",
        "volume", "guide", "book", "illustrated", "edition",
    }]
    if meaningful:
        keyword_query = " ".join(meaningful[:6])
        if author:
            queries.append((f'{keyword_query} inauthor:"{author}"', "keywords+author"))
        queries.append((keyword_query, "keywords-only"))
        queries.append((f'intitle:{meaningful[0]}', "single-title-keyword"))

    if author:
        queries.append((f'inauthor:"{author}"', "author-only"))

    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for query, mode in queries:
        if query not in seen:
            seen.add(query)
            deduped.append((query, mode))
    return deduped


def parse_openlibrary_doc(doc: dict[str, Any]) -> dict[str, Any] | None:
    isbn13, isbn10, all_codes = canonical_isbns([str(value) for value in (doc.get("isbn") or [])])
    if not (isbn13 or isbn10):
        return None
    years = [first_year(value) for value in (doc.get("publish_year") or [])]
    latest_year = max((year for year in years if year), default=first_year(doc.get("first_publish_year")))
    publishers = [str(value) for value in (doc.get("publisher") or [])]
    work_key = str(doc.get("key") or "")
    return {
        "source": "Open Library",
        "source_url": "https://openlibrary.org" + work_key if work_key else "",
        "title": str(doc.get("title") or ""),
        "subtitle": "",
        "authors": [str(value) for value in (doc.get("author_name") or [])],
        "publisher": publishers[-1] if publishers else "",
        "year": latest_year,
        "isbn13": isbn13,
        "isbn10": isbn10,
        "all_isbns": all_codes[:20],
    }


def openlibrary_search(book: dict[str, Any], title_only: bool = False) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "title": clean_query_text(book.get("title")),
        "limit": 30,
        "fields": "key,title,author_name,publisher,first_publish_year,publish_year,isbn",
    }
    author = clean_query_text(primary_author(book.get("author")))
    if author and not title_only:
        params["author"] = author

    payload = http_json(OPEN_LIBRARY_URL, params, "openlibrary") or {}
    candidates: list[dict[str, Any]] = []
    for doc in payload.get("docs", []) or []:
        candidate = parse_openlibrary_doc(doc)
        if candidate:
            candidates.append(candidate)
    return candidates


def candidate_metrics(book: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    title_score = similarity(book.get("title"), candidate.get("title"))
    author_score = author_similarity(book.get("author"), candidate.get("authors") or [])
    publisher_score = (
        similarity(book.get("publisher"), candidate.get("publisher"))
        if useful_publisher(book.get("publisher"))
        else 0.0
    )

    catalog_year = first_year(book.get("year"))
    candidate_year = first_year(candidate.get("year"))
    year_score = 0.0
    if catalog_year and candidate_year:
        difference = abs(catalog_year - candidate_year)
        year_score = 1.0 if difference == 0 else 0.86 if difference == 1 else 0.60 if difference <= 3 else 0.25 if difference <= 10 else 0.0

    if useful_author(book.get("author")):
        score = 0.69 * title_score + 0.25 * author_score
        score += 0.04 * publisher_score if useful_publisher(book.get("publisher")) else 0.03
        score += 0.02 * year_score if catalog_year else 0.03
    else:
        score = 0.88 * title_score
        score += 0.07 * publisher_score if useful_publisher(book.get("publisher")) else 0.05
        score += 0.05 * year_score if catalog_year else 0.02

    if len(tokens(book.get("title"))) <= 2 and not useful_author(book.get("author")):
        score -= 0.10

    return {
        "score": max(0.0, min(1.0, score)),
        "title_score": title_score,
        "author_score": author_score,
        "publisher_score": publisher_score,
        "year_score": year_score,
    }


def rank_candidates(book: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        identity = candidate.get("isbn13") or candidate.get("isbn10")
        if identity:
            unique[str(identity)] = candidate

    scored = [{**candidate, **candidate_metrics(book, candidate)} for candidate in unique.values()]
    return sorted(
        scored,
        key=lambda item: (
            item.get("score", 0.0),
            first_year(item.get("year")) or 0,
            item.get("source") == "Google Books",
        ),
        reverse=True,
    )


def choose_assignment(book: dict[str, Any], candidates: list[dict[str, Any]], mode: str) -> tuple[dict[str, Any] | None, str, str]:
    ranked = rank_candidates(book, candidates)
    if not ranked:
        return None, "Very Low", "No candidate"

    catalog_year = first_year(book.get("year"))
    has_author = useful_author(book.get("author"))
    has_publisher = useful_publisher(book.get("publisher"))

    anchored: list[dict[str, Any]] = []
    if catalog_year or has_publisher:
        for item in ranked:
            author_ok = item["author_score"] >= 0.65 or not has_author
            year_ok = item["year_score"] >= 0.60 if catalog_year else False
            publisher_ok = item["publisher_score"] >= 0.62 if has_publisher else False
            if item["title_score"] >= 0.88 and author_ok and (year_ok or publisher_ok):
                anchored.append(item)
    if anchored:
        best = max(anchored, key=lambda item: (item["score"], item["year_score"], item["publisher_score"]))
        confidence = "High" if best["score"] >= 0.90 and best["title_score"] >= 0.94 else "Medium"
        return best, confidence, "Exact/probable catalogue edition"

    plausible: list[dict[str, Any]] = []
    for item in ranked:
        author_ok = item["author_score"] >= 0.50 or not has_author
        if item["title_score"] >= 0.84 and author_ok:
            plausible.append(item)
    if plausible:
        latest = max(plausible, key=lambda item: (first_year(item.get("year")) or 0, item["score"]))
        confidence = "Medium" if latest["score"] >= 0.86 and latest["title_score"] >= 0.92 else "Low"
        return latest, confidence, "Latest known ISBN-bearing edition of matched work"

    title_matches = [item for item in ranked if item["title_score"] >= 0.68]
    if title_matches:
        latest = max(title_matches, key=lambda item: (first_year(item.get("year")) or 0, item["score"]))
        return latest, "Very Low", f"Latest plausible title match ({mode}); exact work/edition unconfirmed"

    latest = max(ranked, key=lambda item: (first_year(item.get("year")) or 0, item["score"]))
    return latest, "Very Low", f"Closest real ISBN result ({mode}); likely not the exact work"


def synthetic_isbn13(book_id: str) -> str:
    number = int(re.sub(r"\D", "", book_id) or "0")
    stem = "9780000" + f"{number:05d}"
    total = sum((1 if index % 2 == 0 else 3) * int(char) for index, char in enumerate(stem))
    return stem + str((10 - total % 10) % 10)


def result_from_candidate(book: dict[str, Any], candidate: dict[str, Any], confidence: str, basis: str, query_mode: str) -> dict[str, Any]:
    isbn13 = str(candidate.get("isbn13") or "")
    isbn10 = str(candidate.get("isbn10") or "")
    if not isbn13 and isbn10:
        isbn13 = isbn10_to_13(isbn10) or ""
    if not isbn10 and isbn13:
        isbn10 = isbn13_to_10(isbn13) or ""

    return {
        "id": book["id"],
        "isbn13": isbn13,
        "isbn10": isbn10,
        "isbn_confidence": confidence,
        "isbn_status": basis,
        "candidate_isbn13": isbn13,
        "candidate_isbn10": isbn10,
        "matched_title": str(candidate.get("title") or ""),
        "matched_authors": "; ".join(str(value) for value in (candidate.get("authors") or [])),
        "matched_publisher": str(candidate.get("publisher") or ""),
        "matched_year": candidate.get("year") or "",
        "match_score": round(float(candidate.get("score") or 0.0), 3),
        "isbn_source": str(candidate.get("source") or ""),
        "source_url": str(candidate.get("source_url") or ""),
        "alternative_isbns": "; ".join(str(value) for value in (candidate.get("all_isbns") or [])[:20]),
        "assignment_basis": basis,
        "query_mode": query_mode,
        "synthetic": False,
    }


def assign_book(book: dict[str, Any]) -> dict[str, Any]:
    accumulated: list[dict[str, Any]] = []

    for query, mode in google_queries(book):
        candidates = google_search(query, newest=True)
        accumulated.extend(candidates)
        chosen, confidence, basis = choose_assignment(book, accumulated, mode)
        if chosen:
            if confidence in ("High", "Medium", "Low"):
                return result_from_candidate(book, chosen, confidence, basis, mode)
            if mode in ("title-only", "keywords-only", "single-title-keyword", "author-only"):
                return result_from_candidate(book, chosen, confidence, basis, mode)

    for title_only in (False, True):
        mode = "openlibrary-title-only" if title_only else "openlibrary-title+author"
        candidates = openlibrary_search(book, title_only=title_only)
        accumulated.extend(candidates)
        chosen, confidence, basis = choose_assignment(book, accumulated, mode)
        if chosen:
            return result_from_candidate(book, chosen, confidence, basis, mode)

    title_tokens = [word for word in norm(book.get("title")).split() if len(word) >= 4]
    broad_query = " ".join(title_tokens[:3]) or "books"
    candidates = google_search(broad_query, newest=True)
    chosen, confidence, basis = choose_assignment(book, accumulated + candidates, "broad-keyword-fallback")
    if chosen:
        return result_from_candidate(book, chosen, "Very Low", basis, "broad-keyword-fallback")

    isbn13 = synthetic_isbn13(book["id"])
    return {
        "id": book["id"],
        "isbn13": isbn13,
        "isbn10": isbn13_to_10(isbn13) or "",
        "isbn_confidence": "Synthetic",
        "isbn_status": "Synthetic placeholder only; no real ISBN result was found",
        "candidate_isbn13": isbn13,
        "candidate_isbn10": isbn13_to_10(isbn13) or "",
        "matched_title": "",
        "matched_authors": "",
        "matched_publisher": "",
        "matched_year": "",
        "match_score": 0.0,
        "isbn_source": "Synthetic placeholder",
        "source_url": "",
        "alternative_isbns": "",
        "assignment_basis": "Synthetic placeholder; not a registered ISBN",
        "query_mode": "synthetic",
        "synthetic": True,
    }


def main() -> None:
    books = load_books()
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for book in books:
        key = (
            norm(book.get("title")),
            norm(primary_author(book.get("author"))),
            norm(book.get("publisher")),
        )
        unique.setdefault(key, book)

    print(f"Loaded {len(books)} rows; assigning ISBNs for {len(unique)} unique catalogue identities", flush=True)

    result_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {executor.submit(assign_book, book): key for key, book in unique.items()}
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            key = future_map[future]
            try:
                result_cache[key] = future.result()
            except Exception as exc:
                print(f"Assignment failed for {key}: {exc}", flush=True)
                fallback = dict(unique[key])
                result_cache[key] = assign_book(fallback)
            if completed % 50 == 0 or completed == len(future_map):
                print(f"Assigned {completed}/{len(future_map)} unique identities", flush=True)

    results: list[dict[str, Any]] = []
    for book in books:
        key = (
            norm(book.get("title")),
            norm(primary_author(book.get("author"))),
            norm(book.get("publisher")),
        )
        template = dict(result_cache[key])
        template["id"] = book["id"]
        results.append(template)

    for old_path in RESULTS_DIR.glob("*.json"):
        old_path.unlink()

    for start in range(0, len(results), 100):
        chunk = results[start:start + 100]
        path = RESULTS_DIR / f"isbn_results_{start // 100:02d}.json"
        path.write_text(json.dumps(chunk, ensure_ascii=False, indent=2), encoding="utf-8")

    confidence_counts: dict[str, int] = {}
    basis_counts: dict[str, int] = {}
    for result in results:
        confidence_counts[result["isbn_confidence"]] = confidence_counts.get(result["isbn_confidence"], 0) + 1
        basis_counts[result["assignment_basis"]] = basis_counts.get(result["assignment_basis"], 0) + 1

    summary = {
        "total": len(results),
        "assigned_isbn_rows": sum(1 for row in results if row.get("isbn13") or row.get("isbn10")),
        "synthetic_rows": sum(1 for row in results if row.get("synthetic")),
        "confidence_counts": confidence_counts,
        "assignment_basis_counts": basis_counts,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
