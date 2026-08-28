#!/usr/bin/env python3
"""Fast, deliberately aggressive ISBN assignment for the 520 blank rows."""
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

from enrich import author_similarity, canonical_isbns, first_year, isbn10_to_13, isbn13_to_10, norm, primary_author, similarity, useful_author

ROOT = Path(__file__).resolve().parent
BOOKS_PATH = ROOT / "data" / "missing_books_forced.json"
OUT_DIR = ROOT / "results_missing_fast"
OUT_DIR.mkdir(parents=True, exist_ok=True)
GOOGLE = "https://www.googleapis.com/books/v1/volumes"
OPENLIBRARY = "https://openlibrary.org/search.json"
UA = "JoshElgarBookCatalogueISBN/2.1 (forced latest-edition fallback)"
CACHE: dict[str, dict[str, Any] | None] = {}
LOCK = threading.Lock()


def fetch_json(url: str, params: dict[str, Any]) -> dict[str, Any] | None:
    full = url + "?" + urllib.parse.urlencode(params, doseq=True)
    with LOCK:
        if full in CACHE:
            return CACHE[full]
    for attempt in range(5):
        req = urllib.request.Request(full, headers={"User-Agent": UA, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                value = json.loads(response.read().decode("utf-8"))
                with LOCK:
                    CACHE[full] = value
                return value
        except urllib.error.HTTPError as exc:
            if exc.code not in (429, 500, 502, 503, 504):
                break
            time.sleep((1.7 ** attempt) + random.random())
        except Exception:
            time.sleep((1.7 ** attempt) + random.random())
    with LOCK:
        CACHE[full] = None
    return None


def clean_author(value: Any) -> str:
    text = str(primary_author(value) or "")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\b(editors?|compiler|translator|unknown|uncertain)\b", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip(" /;,.")


def parse_google(payload: dict[str, Any] | None) -> list[dict[str, Any]]:
    output = []
    for item in (payload or {}).get("items", []) or []:
        info = item.get("volumeInfo") or {}
        codes = [str(v.get("identifier") or "") for v in info.get("industryIdentifiers", []) or []]
        isbn13, isbn10, all_codes = canonical_isbns(codes)
        if not (isbn13 or isbn10):
            continue
        output.append({
            "title": str(info.get("title") or ""),
            "authors": [str(v) for v in info.get("authors", []) or []],
            "publisher": str(info.get("publisher") or ""),
            "year": first_year(info.get("publishedDate")),
            "isbn13": isbn13,
            "isbn10": isbn10,
            "all": all_codes,
            "url": str(info.get("infoLink") or item.get("selfLink") or ""),
            "source": "Google Books",
        })
    return output


def google_query(q: str) -> list[dict[str, Any]]:
    return parse_google(fetch_json(GOOGLE, {"q": q, "maxResults": 40, "orderBy": "newest", "printType": "books", "projection": "full"}))


def openlibrary_query(book: dict[str, Any]) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "title": str(book.get("title") or ""),
        "limit": 20,
        "fields": "key,title,author_name,publisher,first_publish_year,publish_year,isbn",
    }
    author = clean_author(book.get("author"))
    if author:
        params["author"] = author
    payload = fetch_json(OPENLIBRARY, params) or {}
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
            "isbn13": isbn13,
            "isbn10": isbn10,
            "all": all_codes,
            "url": "https://openlibrary.org" + str(doc.get("key") or ""),
            "source": "Open Library",
        })
    return output


def score(book: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, float, float]:
    title_score = similarity(book.get("title"), candidate.get("title"))
    author_score = author_similarity(book.get("author"), candidate.get("authors") or [])
    combined = 0.77 * title_score + (0.20 * author_score if useful_author(book.get("author")) else 0.10) + 0.03
    return min(1.0, combined), title_score, author_score


def choose(book: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, str, str, float]:
    unique = {}
    for candidate in candidates:
        identity = candidate.get("isbn13") or candidate.get("isbn10")
        if identity:
            unique[str(identity)] = candidate
    ranked = []
    for candidate in unique.values():
        combined, title_score, author_score = score(book, candidate)
        ranked.append((candidate, combined, title_score, author_score))
    if not ranked:
        return None, "Very Low", "No result", 0.0

    author_known = useful_author(book.get("author"))
    plausible = [row for row in ranked if row[2] >= 0.78 and (row[3] >= 0.42 or not author_known)]
    if plausible:
        chosen = max(plausible, key=lambda row: (first_year(row[0].get("year")) or 0, row[1]))
        confidence = "Medium" if chosen[2] >= 0.93 and (chosen[3] >= 0.70 or not author_known) else "Low"
        return chosen[0], confidence, "Latest ISBN-bearing edition of matched work", chosen[1]

    title_related = [row for row in ranked if row[2] >= 0.55]
    if title_related:
        chosen = max(title_related, key=lambda row: (first_year(row[0].get("year")) or 0, row[1]))
        return chosen[0], "Very Low", "Latest plausible title match; work/edition uncertain", chosen[1]

    chosen = max(ranked, key=lambda row: (first_year(row[0].get("year")) or 0, row[1]))
    return chosen[0], "Very Low", "Closest real ISBN result; likely not the exact work", chosen[1]


def synthetic(book_id: str) -> str:
    number = int(re.sub(r"\D", "", book_id) or "0")
    stem = "9780000" + f"{number:05d}"
    check = (10 - sum((1 if i % 2 == 0 else 3) * int(c) for i, c in enumerate(stem)) % 10) % 10
    return stem + str(check)


def assign(book: dict[str, Any]) -> dict[str, Any]:
    title = str(book.get("title") or "").strip()
    author = clean_author(book.get("author"))
    candidates: list[dict[str, Any]] = []

    if title and author:
        candidates.extend(google_query(f'intitle:"{title}" inauthor:"{author}"'))
    if not candidates and title:
        candidates.extend(google_query(f'intitle:"{title}"'))
    if not candidates:
        candidates.extend(openlibrary_query(book))
    if not candidates:
        words = [w for w in norm(title).split() if len(w) >= 4 and w not in {"book", "guide", "volume", "illustrated"}]
        broad = " ".join(words[:4]) or author or "history"
        candidates.extend(google_query(broad))

    selected, confidence, basis, match_score = choose(book, candidates)
    if selected is None:
        isbn13 = synthetic(str(book["id"]))
        return {
            "id": book["id"], "isbn13": isbn13, "isbn10": isbn13_to_10(isbn13) or "",
            "isbn_confidence": "Synthetic", "isbn_status": "Synthetic placeholder; not a registered ISBN",
            "matched_title": "", "matched_authors": "", "matched_publisher": "", "matched_year": "",
            "match_score": 0.0, "isbn_source": "Synthetic placeholder", "source_url": "",
            "alternative_isbns": "", "assignment_basis": "No online result after aggressive search", "synthetic": True,
        }

    isbn13 = str(selected.get("isbn13") or "")
    isbn10 = str(selected.get("isbn10") or "")
    if not isbn13 and isbn10:
        isbn13 = isbn10_to_13(isbn10) or ""
    if not isbn10 and isbn13:
        isbn10 = isbn13_to_10(isbn13) or ""
    return {
        "id": book["id"], "isbn13": isbn13, "isbn10": isbn10,
        "isbn_confidence": confidence, "isbn_status": basis,
        "matched_title": selected.get("title") or "",
        "matched_authors": "; ".join(selected.get("authors") or []),
        "matched_publisher": selected.get("publisher") or "", "matched_year": selected.get("year") or "",
        "match_score": round(match_score, 3), "isbn_source": selected.get("source") or "",
        "source_url": selected.get("url") or "", "alternative_isbns": "; ".join((selected.get("all") or [])[:20]),
        "assignment_basis": basis, "synthetic": False,
    }


def main() -> None:
    books = json.loads(BOOKS_PATH.read_text(encoding="utf-8"))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_map = {executor.submit(assign, book): book for book in books}
        for done, future in enumerate(concurrent.futures.as_completed(future_map), 1):
            book = future_map[future]
            try:
                results.append(future.result())
            except Exception as exc:
                print(f"Failed {book['id']}: {exc}", flush=True)
                results.append(assign(book))
            if done % 50 == 0:
                print(f"Assigned {done}/{len(books)}", flush=True)
    results.sort(key=lambda row: int(re.sub(r"\D", "", str(row["id"])) or "0"))
    for old in OUT_DIR.glob("*.json"):
        old.unlink()
    for start in range(0, len(results), 100):
        (OUT_DIR / f"results_{start // 100:02d}.json").write_text(json.dumps(results[start:start+100], ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "total": len(results),
        "assigned": sum(1 for row in results if row.get("isbn13") or row.get("isbn10")),
        "synthetic": sum(1 for row in results if row.get("synthetic")),
        "confidence": {value: sum(1 for row in results if row.get("isbn_confidence") == value) for value in ["Medium", "Low", "Very Low", "Synthetic"]},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
