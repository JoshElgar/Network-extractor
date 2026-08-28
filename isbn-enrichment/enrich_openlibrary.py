#!/usr/bin/env python3
"""Fast, conservative Open Library ISBN enrichment for the 806-book catalogue.

The script searches in small batches, then performs targeted edition lookups only
when needed. Exact/probable ISBNs are separated from unconfirmed candidates.
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
    likely_pre_isbn,
    load_books,
    norm,
    primary_author,
    similarity,
    tokens,
    useful_author,
)

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results_openlibrary"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SEARCH_URL = "https://openlibrary.org/search.json"
CONTACT_EMAIL = "joshelgar@gmail.com"
USER_AGENT = f"JoshElgarBookCatalogueISBN/1.0 ({CONTACT_EMAIL})"
FIELDS = ",".join([
    "key", "title", "author_name", "first_publish_year",
    "editions", "editions.key", "editions.title", "editions.subtitle",
    "editions.publisher", "editions.publish_date", "editions.publish_year",
    "editions.isbn_10", "editions.isbn_13", "editions.language",
])
_RATE_LOCK = threading.Lock()
_LAST_REQUEST = 0.0
MIN_INTERVAL_SECONDS = 0.36  # identified Open Library clients are allowed 3 req/s


def paced() -> None:
    global _LAST_REQUEST
    with _RATE_LOCK:
        now = time.monotonic()
        wait = MIN_INTERVAL_SECONDS - (now - _LAST_REQUEST)
        if wait > 0:
            time.sleep(wait)
        _LAST_REQUEST = time.monotonic()


def http_json(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    full_url = url if not params else url + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        paced()
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
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt) + random.random()
                time.sleep(delay)
                continue
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep((2 ** attempt) + random.random())
    return None


def escape_phrase(value: Any) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"').strip()


def useful_publisher(value: Any) -> bool:
    text = norm(value)
    if not text:
        return False
    rejected = ("uncertain", "unknown", "edition imprint", "edition uncertain")
    return not any(term in text for term in rejected)


def query_publisher(value: Any) -> str:
    if not useful_publisher(value):
        return ""
    raw = str(value).split("/")[0].strip()
    return raw if len(norm(raw)) >= 2 else ""


def book_key(book: dict[str, Any]) -> tuple[str, str, str]:
    return (norm(book.get("title")), norm(primary_author(book.get("author"))), norm(query_publisher(book.get("publisher"))))


def query_clause(book: dict[str, Any], include_publisher: bool = True) -> str:
    parts = [f'title:"{escape_phrase(book.get("title"))}"']
    author = primary_author(book.get("author"))
    if author:
        parts.append(f'author:"{escape_phrase(author)}"')
    publisher = query_publisher(book.get("publisher")) if include_publisher else ""
    if publisher:
        parts.append(f'publisher:"{escape_phrase(publisher)}"')
    return "(" + " AND ".join(parts) + ")"


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def candidate_from_work(doc: dict[str, Any]) -> dict[str, Any]:
    edition_docs = ((doc.get("editions") or {}).get("docs") or [])
    edition = edition_docs[0] if edition_docs else {}
    isbn_values = listify(edition.get("isbn_13")) + listify(edition.get("isbn_10"))
    isbn13, isbn10, all_codes = canonical_isbns([str(value) for value in isbn_values])
    publishers = listify(edition.get("publisher"))
    publish_dates = listify(edition.get("publish_date")) or listify(edition.get("publish_year"))
    edition_key = str(edition.get("key") or "")
    return {
        "source": "Open Library",
        "work_key": str(doc.get("key") or ""),
        "edition_key": edition_key,
        "source_url": "https://openlibrary.org" + (edition_key or str(doc.get("key") or "")),
        "work_title": str(doc.get("title") or ""),
        "title": str(edition.get("title") or doc.get("title") or ""),
        "subtitle": str(edition.get("subtitle") or ""),
        "authors": listify(doc.get("author_name")),
        "publisher": str(publishers[0]) if publishers else "",
        "year": first_year(publish_dates[0]) if publish_dates else first_year(doc.get("first_publish_year")),
        "isbn13": isbn13,
        "isbn10": isbn10,
        "all_isbns": all_codes,
    }


def score_candidate(book: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    title_score = max(similarity(book.get("title"), candidate.get("title")), similarity(book.get("title"), candidate.get("work_title")))
    author_score = author_similarity(book.get("author"), candidate.get("authors") or [])
    publisher_score = similarity(query_publisher(book.get("publisher")), candidate.get("publisher")) if useful_publisher(book.get("publisher")) else 0.0
    catalog_year = first_year(book.get("year"))
    candidate_year = first_year(candidate.get("year"))
    year_score = 0.0
    if catalog_year and candidate_year:
        difference = abs(catalog_year - candidate_year)
        year_score = 1.0 if difference == 0 else 0.85 if difference == 1 else 0.55 if difference <= 3 else 0.15 if difference <= 10 else 0.0

    weights: list[tuple[float, float]] = [(0.66, title_score)]
    if useful_author(book.get("author")):
        weights.append((0.24, author_score))
    else:
        weights[0] = (0.82, title_score)
    if useful_publisher(book.get("publisher")):
        weights.append((0.08, publisher_score))
    if catalog_year:
        weights.append((0.02, year_score))
    score = sum(weight * value for weight, value in weights) / sum(weight for weight, _ in weights)
    if len(tokens(book.get("title"))) <= 2 and not useful_author(book.get("author")):
        score -= 0.14
    return {
        "score": max(0.0, min(1.0, score)),
        "title_score": title_score,
        "author_score": author_score,
        "publisher_score": publisher_score,
        "year_score": year_score,
    }


def search_group(group: list[dict[str, Any]], include_publisher: bool = True) -> list[dict[str, Any]]:
    query = " OR ".join(query_clause(book, include_publisher=include_publisher) for book in group)
    payload = http_json(SEARCH_URL, {"q": query, "lang": "en", "fields": FIELDS, "limit": min(100, max(25, len(group) * 12))})
    return [candidate_from_work(doc) for doc in (payload or {}).get("docs", []) or []]


def targeted_search(book: dict[str, Any], include_publisher: bool) -> list[dict[str, Any]]:
    payload = http_json(SEARCH_URL, {"q": query_clause(book, include_publisher=include_publisher), "lang": "en", "fields": FIELDS, "limit": 8})
    return [candidate_from_work(doc) for doc in (payload or {}).get("docs", []) or []]


def fetch_edition(candidate: dict[str, Any]) -> dict[str, Any]:
    edition_key = candidate.get("edition_key") or ""
    if not edition_key:
        return candidate
    payload = http_json("https://openlibrary.org" + edition_key + ".json") or {}
    isbn_values = listify(payload.get("isbn_13")) + listify(payload.get("isbn_10")) + listify(payload.get("isbn"))
    isbn13, isbn10, all_codes = canonical_isbns([str(value) for value in isbn_values])
    publishers = listify(payload.get("publishers"))
    updated = dict(candidate)
    updated.update({
        "title": str(payload.get("title") or candidate.get("title") or ""),
        "publisher": str(publishers[0]) if publishers else candidate.get("publisher", ""),
        "year": first_year(payload.get("publish_date")) or candidate.get("year"),
        "isbn13": isbn13 or candidate.get("isbn13", ""),
        "isbn10": isbn10 or candidate.get("isbn10", ""),
        "all_isbns": all_codes or candidate.get("all_isbns", []),
    })
    return updated


def rank(book: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        identity = candidate.get("edition_key") or candidate.get("work_key") or json.dumps(candidate, sort_keys=True)
        unique[identity] = candidate
    scored = [{**candidate, **score_candidate(book, candidate)} for candidate in unique.values()]
    return sorted(scored, key=lambda item: item["score"], reverse=True)


def confidence_for(book: dict[str, Any], candidate: dict[str, Any]) -> tuple[str, str]:
    score = candidate["score"]
    title_score = candidate["title_score"]
    author_score = candidate["author_score"]
    publisher_score = candidate["publisher_score"]
    year_score = candidate["year_score"]
    author_ok = author_score >= 0.72 or not useful_author(book.get("author"))
    publisher_known = useful_publisher(book.get("publisher"))
    year_known = bool(first_year(book.get("year")))
    publisher_anchor = publisher_known and publisher_score >= 0.72
    year_anchor = year_known and year_score >= 0.85
    generic = len(tokens(book.get("title"))) <= 2 and not useful_author(book.get("author"))

    if candidate.get("isbn13") or candidate.get("isbn10"):
        if score >= 0.90 and title_score >= 0.94 and author_ok and (publisher_anchor or year_anchor) and not generic:
            return "High", "Exact or strongly anchored edition match"
        if score >= 0.80 and title_score >= 0.88 and author_ok and (publisher_anchor or year_anchor) and not generic:
            return "Medium", "Probable edition match"
        if score >= 0.86 and title_score >= 0.94 and author_ok and not generic:
            return "Low", "Work match; exact edition unconfirmed"
    return "None", "No safe ISBN assignment"


def result_for(book: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scored = rank(book, candidates)
    best = scored[0] if scored else None
    result: dict[str, Any] = {
        "id": book["id"], "isbn13": "", "isbn10": "", "isbn_confidence": "None",
        "isbn_status": "No ISBN found", "candidate_isbn13": "", "candidate_isbn10": "",
        "matched_title": "", "matched_authors": "", "matched_publisher": "", "matched_year": "",
        "match_score": "", "isbn_source": "Open Library", "source_url": "", "alternative_isbns": "",
    }
    if not best:
        result["isbn_status"] = "No ISBN likely: edition predates ISBN" if likely_pre_isbn(book) else "No ISBN found"
        return result
    confidence, status = confidence_for(book, best)
    result.update({
        "candidate_isbn13": best.get("isbn13", ""), "candidate_isbn10": best.get("isbn10", ""),
        "matched_title": best.get("title", ""), "matched_authors": "; ".join(str(x) for x in best.get("authors", []) or []),
        "matched_publisher": best.get("publisher", ""), "matched_year": best.get("year") or "",
        "match_score": round(best.get("score", 0.0), 3), "source_url": best.get("source_url", ""),
        "alternative_isbns": "; ".join(best.get("all_isbns", [])[:10]), "isbn_confidence": confidence,
        "isbn_status": status,
    })
    if confidence in ("High", "Medium"):
        result["isbn13"] = best.get("isbn13", "")
        result["isbn10"] = best.get("isbn10", "")
    elif confidence == "None" and likely_pre_isbn(book):
        result["isbn_status"] = "No safe ISBN assignment; catalogue edition likely predates ISBN"
    return result


def main() -> None:
    books = load_books()
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for book in books:
        unique.setdefault(book_key(book), book)
    unique_books = list(unique.values())
    groups = [unique_books[index:index + 5] for index in range(0, len(unique_books), 5)]
    print(f"Loaded {len(books)} rows, {len(unique_books)} unique searches, {len(groups)} batches", flush=True)

    candidates_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = {book_key(book): [] for book in unique_books}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(search_group, group, True): group for group in groups}
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            group = future_map[future]
            try:
                candidates = future.result()
            except Exception as exc:
                print(f"Batch error: {exc}", flush=True)
                candidates = []
            for book in group:
                ranked = rank(book, candidates)
                candidates_by_key[book_key(book)].extend(ranked[:5])
            if completed % 25 == 0 or completed == len(groups):
                print(f"Completed batch searches {completed}/{len(groups)}", flush=True)

    # Retry weak/missing matches without a publisher constraint. This also catches
    # catalog publisher abbreviations that Open Library normalizes differently.
    retry_books: list[dict[str, Any]] = []
    for book in unique_books:
        ranked = rank(book, candidates_by_key[book_key(book)])
        if not ranked or ranked[0]["score"] < 0.78:
            retry_books.append(book)
    print(f"Targeted retries: {len(retry_books)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(targeted_search, book, False): book for book in retry_books}
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            book = future_map[future]
            try:
                candidates_by_key[book_key(book)].extend(future.result())
            except Exception as exc:
                print(f"Retry error {book['id']}: {exc}", flush=True)
            if completed % 50 == 0 or completed == len(future_map):
                print(f"Completed targeted retries {completed}/{len(future_map)}", flush=True)

    # Fetch edition JSON only for strong work matches where the search result did
    # not expose an ISBN. This limits extra API calls and improves edition metadata.
    edition_fetches: dict[str, tuple[tuple[str, str, str], dict[str, Any]]] = {}
    for book in unique_books:
        key = book_key(book)
        ranked = rank(book, candidates_by_key[key])
        if not ranked:
            continue
        best = ranked[0]
        if best["score"] >= 0.74 and best.get("edition_key") and not (best.get("isbn13") or best.get("isbn10")) and not likely_pre_isbn(book):
            edition_fetches.setdefault(best["edition_key"], (key, best))
    print(f"Edition detail fetches: {len(edition_fetches)}", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(fetch_edition, candidate): (edition_key, key) for edition_key, (key, candidate) in edition_fetches.items()}
        for completed, future in enumerate(concurrent.futures.as_completed(future_map), start=1):
            _, key = future_map[future]
            try:
                candidates_by_key[key].append(future.result())
            except Exception as exc:
                print(f"Edition fetch error: {exc}", flush=True)
            if completed % 50 == 0 or completed == len(future_map):
                print(f"Completed edition fetches {completed}/{len(future_map)}", flush=True)

    results = [result_for(book, candidates_by_key.get(book_key(book), [])) for book in books]
    for index in range(0, len(results), 100):
        chunk = results[index:index + 100]
        (RESULTS_DIR / f"isbn_results_{index // 100:02d}.json").write_text(json.dumps(chunk, ensure_ascii=False, indent=2), encoding="utf-8")
    status_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    for row in results:
        status_counts[row["isbn_status"]] = status_counts.get(row["isbn_status"], 0) + 1
        confidence_counts[row["isbn_confidence"]] = confidence_counts.get(row["isbn_confidence"], 0) + 1
    summary = {
        "total": len(results),
        "with_primary_isbn": sum(bool(row["isbn13"] or row["isbn10"]) for row in results),
        "with_any_candidate": sum(bool(row["candidate_isbn13"] or row["candidate_isbn10"]) for row in results),
        "confidence_counts": confidence_counts,
        "status_counts": status_counts,
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
