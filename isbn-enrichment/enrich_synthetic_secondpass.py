#!/usr/bin/env python3
"""Second-pass real ISBN search for rows that failed the first aggressive run."""
from __future__ import annotations

import concurrent.futures
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from enrich import author_similarity, canonical_isbns, first_year, isbn10_to_13, isbn13_to_10, load_books, norm, primary_author, similarity, useful_author

ROOT = Path(__file__).resolve().parent
IDS_PATH = ROOT / "data" / "synthetic_ids_208.json"
OUT_DIR = ROOT / "results_synthetic_secondpass"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CHUNK_INDEX = int(os.environ.get("CHUNK_INDEX", "0"))
CHUNK_COUNT = int(os.environ.get("CHUNK_COUNT", "4"))
GOOGLE = "https://www.googleapis.com/books/v1/volumes"
OPENLIBRARY = "https://openlibrary.org/search.json"
UA = "JoshElgarBookCatalogueISBN/2.2 (second-pass real ISBN search)"
CACHE: dict[str, dict[str, Any] | None] = {}
LOCK = threading.Lock()


def fetch(url: str, params: dict[str, Any]) -> dict[str, Any] | None:
    full = url + "?" + urllib.parse.urlencode(params, doseq=True)
    with LOCK:
        if full in CACHE:
            return CACHE[full]
    for attempt in range(4):
        req = urllib.request.Request(full, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=35) as response:
                value = json.loads(response.read().decode("utf-8"))
                with LOCK:
                    CACHE[full] = value
                return value
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                break
            time.sleep((2 ** attempt) + random.random())
        except Exception:
            time.sleep((2 ** attempt) + random.random())
    with LOCK:
        CACHE[full] = None
    return None


def clean_author(value: Any) -> str:
    text = str(primary_author(value) or "")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(editors?|compiler|translator|unknown|uncertain)\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" /;,.")


def google(q: str, start: int = 0) -> list[dict[str, Any]]:
    payload = fetch(GOOGLE, {"q": q, "maxResults": 40, "startIndex": start, "printType": "books", "projection": "full"}) or {}
    output = []
    for item in payload.get("items", []) or []:
        info = item.get("volumeInfo") or {}
        isbn13, isbn10, all_codes = canonical_isbns([str(v.get("identifier") or "") for v in info.get("industryIdentifiers", []) or []])
        if not (isbn13 or isbn10):
            continue
        output.append({
            "title": str(info.get("title") or ""),
            "authors": [str(v) for v in info.get("authors", []) or []],
            "publisher": str(info.get("publisher") or ""),
            "year": first_year(info.get("publishedDate")),
            "isbn13": isbn13, "isbn10": isbn10, "all": all_codes,
            "url": str(info.get("infoLink") or item.get("selfLink") or ""), "source": "Google Books",
        })
    return output


def openlibrary(q: str) -> list[dict[str, Any]]:
    payload = fetch(OPENLIBRARY, {"q": q, "limit": 50, "fields": "key,title,author_name,publisher,first_publish_year,publish_year,isbn"}) or {}
    output = []
    for doc in payload.get("docs", []) or []:
        isbn13, isbn10, all_codes = canonical_isbns([str(v) for v in doc.get("isbn", []) or []])
        if not (isbn13 or isbn10):
            continue
        years = [first_year(v) for v in doc.get("publish_year", []) or []]
        output.append({
            "title": str(doc.get("title") or ""),
            "authors": [str(v) for v in doc.get("author_name", []) or []],
            "publisher": str((doc.get("publisher") or [""])[-1]),
            "year": max((v for v in years if v), default=first_year(doc.get("first_publish_year"))),
            "isbn13": isbn13, "isbn10": isbn10, "all": all_codes[:30],
            "url": "https://openlibrary.org" + str(doc.get("key") or ""), "source": "Open Library",
        })
    return output


def metrics(book: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, float, float]:
    title_score = similarity(book.get("title"), candidate.get("title"))
    author_score = author_similarity(book.get("author"), candidate.get("authors") or [])
    combined = 0.76 * title_score + (0.21 * author_score if useful_author(book.get("author")) else 0.10) + 0.03
    return min(1.0, combined), title_score, author_score


def select(book: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, str, float]:
    unique = {}
    for candidate in candidates:
        key = candidate.get("isbn13") or candidate.get("isbn10")
        if key:
            unique[str(key)] = candidate
    ranked = []
    for candidate in unique.values():
        score, title_score, author_score = metrics(book, candidate)
        ranked.append((candidate, score, title_score, author_score))
    if not ranked:
        return None, "Very Low", "No real ISBN result", 0.0

    author_known = useful_author(book.get("author"))
    plausible = [row for row in ranked if row[2] >= 0.72 and (row[3] >= 0.35 or not author_known)]
    if plausible:
        chosen = max(plausible, key=lambda row: (first_year(row[0].get("year")) or 0, row[1]))
        confidence = "Low" if chosen[2] >= 0.86 and (chosen[3] >= 0.55 or not author_known) else "Very Low"
        return chosen[0], confidence, "Latest real ISBN-bearing edition of plausible work match", chosen[1]

    title_related = [row for row in ranked if row[2] >= 0.45]
    if title_related:
        chosen = max(title_related, key=lambda row: (first_year(row[0].get("year")) or 0, row[1]))
        return chosen[0], "Very Low", "Latest loose title match; exact work/edition uncertain", chosen[1]

    author_related = [row for row in ranked if row[3] >= 0.45]
    if author_related:
        chosen = max(author_related, key=lambda row: (first_year(row[0].get("year")) or 0, row[1]))
        return chosen[0], "Very Low", "Latest ISBN by probable author; title does not match closely", chosen[1]

    chosen = max(ranked, key=lambda row: (row[1], first_year(row[0].get("year")) or 0))
    return chosen[0], "Very Low", "Closest real search result; likely not the exact work", chosen[1]


def package(book: dict[str, Any], candidate: dict[str, Any], confidence: str, basis: str, match_score: float) -> dict[str, Any]:
    isbn13 = str(candidate.get("isbn13") or "")
    isbn10 = str(candidate.get("isbn10") or "")
    if not isbn13 and isbn10:
        isbn13 = isbn10_to_13(isbn10) or ""
    if not isbn10 and isbn13:
        isbn10 = isbn13_to_10(isbn13) or ""
    return {
        "id": book["id"], "isbn13": isbn13, "isbn10": isbn10,
        "isbn_confidence": confidence, "isbn_status": basis,
        "matched_title": candidate.get("title") or "", "matched_authors": "; ".join(candidate.get("authors") or []),
        "matched_publisher": candidate.get("publisher") or "", "matched_year": candidate.get("year") or "",
        "match_score": round(match_score, 3), "isbn_source": candidate.get("source") or "",
        "source_url": candidate.get("url") or "", "alternative_isbns": "; ".join((candidate.get("all") or [])[:30]),
        "assignment_basis": basis, "synthetic": False,
    }


def assign(book: dict[str, Any]) -> dict[str, Any]:
    title = str(book.get("title") or "").strip()
    author = clean_author(book.get("author"))
    title_words = [w for w in norm(title).split() if len(w) >= 3 and w not in {"the", "and", "for", "with", "book", "guide", "volume", "illustrated"}]
    candidates: list[dict[str, Any]] = []

    queries = []
    if title and author:
        queries.append(("ol", f'"{title}" {author}'))
        queries.append(("gb", f'"{title}" "{author}"'))
    if title:
        queries.append(("ol", f'"{title}"'))
        queries.append(("gb", title))
        queries.append(("gb", f'intitle:{" ".join(title_words[:6])}'))
    if author:
        queries.append(("gb", f'inauthor:{author}'))
    if title_words:
        queries.append(("ol", " ".join(title_words[:5])))
        queries.append(("gb", " ".join(title_words[:4])))
        queries.append(("gb", title_words[0]))

    for source, query in queries:
        found = openlibrary(query) if source == "ol" else google(query)
        candidates.extend(found)
        selected, confidence, basis, match_score = select(book, candidates)
        if selected and confidence == "Low":
            return package(book, selected, confidence, basis, match_score)
        if selected and len(candidates) >= 20 and query == queries[-1][1]:
            return package(book, selected, confidence, basis, match_score)

    selected, confidence, basis, match_score = select(book, candidates)
    if selected:
        return package(book, selected, confidence, basis, match_score)

    # Guaranteed final real-ISBN fallback. This intentionally uses a broad book
    # search and is marked Very Low because it may be unrelated.
    fallback = google("books") or openlibrary("books")
    selected, _, _, match_score = select(book, fallback)
    if selected:
        return package(book, selected, "Very Low", "Forced real-ISBN fallback; likely unrelated to the physical book", match_score)

    # Only a total catalogue/API outage reaches this point.
    return {
        "id": book["id"], "isbn13": "", "isbn10": "", "isbn_confidence": "Unresolved",
        "isbn_status": "No real ISBN returned after second pass", "matched_title": "", "matched_authors": "",
        "matched_publisher": "", "matched_year": "", "match_score": 0.0, "isbn_source": "",
        "source_url": "", "alternative_isbns": "", "assignment_basis": "No real result", "synthetic": False,
    }


def main() -> None:
    ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
    books_by_id = {str(book["id"]): book for book in load_books()}
    selected_ids = ids[CHUNK_INDEX::CHUNK_COUNT]
    books = [books_by_id[book_id] for book_id in selected_ids]
    print(f"Second-pass chunk {CHUNK_INDEX}: {len(books)} books", flush=True)
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {executor.submit(assign, book): book for book in books}
        for done, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            book = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Failed {book['id']}: {exc}", flush=True)
                results.append(assign(book))
            if done % 10 == 0 or done == len(books):
                print(f"Chunk {CHUNK_INDEX}: {done}/{len(books)}", flush=True)
    results.sort(key=lambda row: int(re.sub(r"\D", "", str(row["id"])) or "0"))
    (OUT_DIR / f"results_chunk_{CHUNK_INDEX}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"chunk": CHUNK_INDEX, "total": len(results), "real": sum(1 for row in results if row.get("isbn13") or row.get("isbn10"))}), flush=True)


if __name__ == "__main__":
    main()
