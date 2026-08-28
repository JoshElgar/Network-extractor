#!/usr/bin/env python3
"""Enrich the home book catalogue with edition-level ISBN candidates.

Google Books is used as the primary edition source. Open Library is a fallback.
The script never invents an ISBN: weak or ambiguous matches are retained only as
candidates, while the primary ISBN fields are populated for medium/high matches.
"""
from __future__ import annotations

import concurrent.futures
import difflib
import json
import random
import re
import threading
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
USER_AGENT = "JoshElgarBookCatalogueISBN/1.0 (GitHub Actions; metadata enrichment)"
GOOGLE_URL = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"
_RATE_LOCKS = {"google": threading.Lock(), "openlibrary": threading.Lock()}
_LAST_REQUEST = {"google": 0.0, "openlibrary": 0.0}
_MIN_INTERVAL = {"google": 0.13, "openlibrary": 0.45}


def paced(source: str) -> None:
    with _RATE_LOCKS[source]:
        now = time.monotonic()
        delay = _MIN_INTERVAL[source] - (now - _LAST_REQUEST[source])
        if delay > 0:
            time.sleep(delay)
        _LAST_REQUEST[source] = time.monotonic()


def http_json(url: str, params: dict[str, Any], source: str) -> dict[str, Any] | None:
    full_url = url + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        paced(source)
        req = urllib.request.Request(full_url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=25) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep((2 ** attempt) + random.random())
                continue
            return None
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            time.sleep((2 ** attempt) + random.random())
    return None


def norm(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().replace("&", " and ")
    text = re.sub(r"\b(vol(?:ume)?|book)\s*[ivxlcdm\d]+\b", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def tokens(value: Any) -> set[str]:
    stop = {"the", "a", "an", "of", "and", "to", "in", "for", "with", "from", "by"}
    return {word for word in norm(value).split() if word not in stop}


def similarity(a: Any, b: Any) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    sequence = difflib.SequenceMatcher(None, na, nb).ratio()
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return sequence
    intersection = len(ta & tb)
    precision = intersection / len(tb)
    recall = intersection / len(ta)
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    containment = intersection / min(len(ta), len(tb))
    return max(sequence, f1, 0.92 * containment)


def useful_author(author: Any) -> bool:
    value = norm(author)
    bad = ("unknown", "uncertain", "editors", "editor uncertain", "author uncertain", "institution")
    return bool(value) and not any(term in value for term in bad)


def primary_author(author: Any) -> str:
    if not useful_author(author):
        return ""
    text = str(author)
    text = re.sub(r"\([^)]*\)", "", text)
    text = re.sub(r"\b(translated|introduced|selected|illustrated|photographs?|foreword|commentary)\s+by\b.*", "", text, flags=re.I)
    first = re.split(r";|\s+/\s+|\s+with\s+", text)[0].strip()
    first = re.sub(r"^(sir|lord|general|captain|hrh|rabbi)\s+", "", first, flags=re.I)
    return first.strip()


def author_similarity(catalog_author: Any, candidate_authors: Iterable[str]) -> float:
    if not useful_author(catalog_author):
        return 0.0
    primary = primary_author(catalog_author)
    best = 0.0
    for candidate in candidate_authors or []:
        best = max(best, similarity(primary, candidate), similarity(catalog_author, candidate))
        a_surname = norm(primary).split()[-1:] or [""]
        c_surname = norm(candidate).split()[-1:] or [""]
        if a_surname[0] and a_surname[0] == c_surname[0]:
            best = max(best, 0.82)
    return best


def first_year(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"(1[4-9]\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None


def isbn10_valid(value: str) -> bool:
    code = re.sub(r"[^0-9Xx]", "", value or "").upper()
    if len(code) != 10:
        return False
    total = 0
    for index, char in enumerate(code):
        digit = 10 if char == "X" and index == 9 else (int(char) if char.isdigit() else -99)
        total += (10 - index) * digit
    return total % 11 == 0


def isbn13_valid(value: str) -> bool:
    code = re.sub(r"\D", "", value or "")
    if len(code) != 13:
        return False
    total = sum((1 if index % 2 == 0 else 3) * int(char) for index, char in enumerate(code[:12]))
    return (10 - total % 10) % 10 == int(code[-1])


def isbn10_to_13(value: str) -> str | None:
    code = re.sub(r"[^0-9Xx]", "", value or "").upper()
    if not isbn10_valid(code):
        return None
    stem = "978" + code[:9]
    total = sum((1 if index % 2 == 0 else 3) * int(char) for index, char in enumerate(stem))
    return stem + str((10 - total % 10) % 10)


def isbn13_to_10(value: str) -> str | None:
    code = re.sub(r"\D", "", value or "")
    if not isbn13_valid(code) or not code.startswith("978"):
        return None
    stem = code[3:12]
    total = sum((10 - index) * int(char) for index, char in enumerate(stem))
    check = (11 - total % 11) % 11
    check_char = "X" if check == 10 else "0" if check == 0 else str(check)
    return stem + check_char


def canonical_isbns(values: Iterable[str]) -> tuple[str, str, list[str]]:
    isbn10 = ""
    isbn13 = ""
    all_codes: list[str] = []
    for raw in values or []:
        code10 = re.sub(r"[^0-9Xx]", "", str(raw)).upper()
        code13 = re.sub(r"\D", "", str(raw))
        if isbn13_valid(code13):
            if code13 not in all_codes:
                all_codes.append(code13)
            isbn13 = isbn13 or code13
            isbn10 = isbn10 or (isbn13_to_10(code13) or "")
        elif isbn10_valid(code10):
            if code10 not in all_codes:
                all_codes.append(code10)
            isbn10 = isbn10 or code10
            isbn13 = isbn13 or (isbn10_to_13(code10) or "")
    return isbn13, isbn10, all_codes


def load_books() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("books_*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted(DATA_DIR.glob("books_*.tsv")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            fields = (line.split("\t") + [""] * 5)[:5]
            records.append({"id": fields[0], "title": fields[1], "author": fields[2], "year": fields[3] or None, "publisher": fields[4] or None, "notes": "", "id_confidence": ""})
    by_id = {record["id"]: record for record in records}
    ordered = sorted(by_id.values(), key=lambda item: int(re.sub(r"\D", "", item["id"])))
    if len(ordered) != 806:
        raise RuntimeError(f"Expected 806 catalogue rows, loaded {len(ordered)}")
    return ordered


def google_query(book: dict[str, Any]) -> list[dict[str, Any]]:
    title = str(book.get("title") or "").replace('"', " ")
    author = primary_author(book.get("author"))
    query = f'intitle:"{title}"'
    if author:
        query += f' inauthor:"{author}"'
    payload = http_json(GOOGLE_URL, {"q": query, "maxResults": 40, "printType": "books"}, "google")
    candidates: list[dict[str, Any]] = []
    for item in (payload or {}).get("items", []) or []:
        info = item.get("volumeInfo") or {}
        identifiers = [entry.get("identifier", "") for entry in info.get("industryIdentifiers", []) or []]
        isbn13, isbn10, all_codes = canonical_isbns(identifiers)
        if not (isbn13 or isbn10):
            continue
        candidates.append({"source": "Google Books", "source_id": item.get("id", ""), "source_url": info.get("infoLink", ""), "title": info.get("title", ""), "subtitle": info.get("subtitle", ""), "authors": info.get("authors", []) or [], "publisher": info.get("publisher", ""), "year": first_year(info.get("publishedDate")), "isbn13": isbn13, "isbn10": isbn10, "all_isbns": all_codes})
    return candidates


def openlibrary_query(book: dict[str, Any]) -> list[dict[str, Any]]:
    params: dict[str, Any] = {"title": book.get("title") or "", "limit": 20, "fields": "key,title,author_name,publisher,first_publish_year,publish_year,isbn,edition_key"}
    author = primary_author(book.get("author"))
    if author:
        params["author"] = author
    payload = http_json(OPEN_LIBRARY_URL, params, "openlibrary")
    candidates: list[dict[str, Any]] = []
    for doc in (payload or {}).get("docs", []) or []:
        isbn13, isbn10, all_codes = canonical_isbns(doc.get("isbn", []) or [])
        if not (isbn13 or isbn10):
            continue
        publishers = doc.get("publisher", []) or []
        years = doc.get("publish_year", []) or []
        candidates.append({"source": "Open Library", "source_id": doc.get("key", ""), "source_url": "https://openlibrary.org" + str(doc.get("key", "")), "title": doc.get("title", ""), "subtitle": "", "authors": doc.get("author_name", []) or [], "publisher": publishers[0] if publishers else "", "year": min(years) if years else doc.get("first_publish_year"), "isbn13": isbn13, "isbn10": isbn10, "all_isbns": all_codes[:12]})
    return candidates


def candidate_score(book: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    title_score = similarity(book.get("title"), candidate.get("title"))
    author_score = author_similarity(book.get("author"), candidate.get("authors") or [])
    publisher_score = similarity(book.get("publisher"), candidate.get("publisher")) if book.get("publisher") else 0.0
    catalog_year = first_year(book.get("year"))
    candidate_year = first_year(candidate.get("year"))
    year_score = 0.0
    if catalog_year and candidate_year:
        difference = abs(catalog_year - candidate_year)
        year_score = 1.0 if difference == 0 else 0.8 if difference == 1 else 0.45 if difference <= 3 else 0.0
    if useful_author(book.get("author")):
        score = 0.65 * title_score + 0.25 * author_score
        score += 0.07 * publisher_score if book.get("publisher") else 0.04
        score += 0.03 * year_score if catalog_year else 0.03
    else:
        score = 0.84 * title_score
        score += 0.10 * publisher_score if book.get("publisher") else 0.03
        score += 0.06 * year_score if catalog_year else 0.03
    if len(tokens(book.get("title"))) <= 2 and not useful_author(book.get("author")):
        score -= 0.12
    if candidate.get("source") == "Open Library":
        score -= 0.04
    score = max(0.0, min(1.0, score))
    return {"score": score, "title_score": title_score, "author_score": author_score, "publisher_score": publisher_score, "year_score": year_score}


def likely_pre_isbn(book: dict[str, Any]) -> bool:
    year = first_year(book.get("year"))
    modern_publishers = {"penguin", "folio society", "bracken", "parragon", "taschen", "oxford university press"}
    publisher = norm(book.get("publisher"))
    return bool(year and year < 1970 and not any(name in publisher for name in modern_publishers))


def match_book(book: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [{**candidate, **candidate_score(book, candidate)} for candidate in candidates]
    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0] if scored else None
    result: dict[str, Any] = {"id": book["id"], "isbn13": "", "isbn10": "", "isbn_confidence": "None", "isbn_status": "No confident match", "candidate_isbn13": "", "candidate_isbn10": "", "matched_title": "", "matched_authors": "", "matched_publisher": "", "matched_year": "", "match_score": "", "isbn_source": "", "source_url": "", "alternative_isbns": ""}
    if not best:
        if "philatelic album" in norm(book.get("title")):
            result["isbn_status"] = "Not a published book / no ISBN expected"
        elif likely_pre_isbn(book):
            result["isbn_status"] = "No ISBN likely: edition predates ISBN"
        else:
            result["isbn_status"] = "No ISBN found"
        return result
    score = best["score"]
    title_score = best["title_score"]
    author_score = best["author_score"]
    publisher_score = best["publisher_score"]
    year_score = best["year_score"]
    generic = len(tokens(book.get("title"))) <= 2 and not useful_author(book.get("author"))
    has_edition_anchor = (book.get("publisher") and publisher_score >= 0.62) or (book.get("year") and year_score >= 0.8)
    if score >= 0.89 and title_score >= 0.94 and (author_score >= 0.72 or not useful_author(book.get("author"))) and has_edition_anchor and not generic:
        confidence = "High"
    elif score >= 0.79 and title_score >= 0.87 and (author_score >= 0.55 or not useful_author(book.get("author"))) and not generic:
        confidence = "Medium"
    elif score >= 0.67 and title_score >= 0.78:
        confidence = "Low"
    else:
        confidence = "None"
    result.update({"candidate_isbn13": best.get("isbn13", ""), "candidate_isbn10": best.get("isbn10", ""), "matched_title": best.get("title", ""), "matched_authors": "; ".join(best.get("authors", []) or []), "matched_publisher": best.get("publisher", ""), "matched_year": best.get("year") or "", "match_score": round(score, 3), "isbn_source": best.get("source", ""), "source_url": best.get("source_url", ""), "alternative_isbns": "; ".join((best.get("all_isbns") or [])[:8]), "isbn_confidence": confidence})
    if confidence in ("High", "Medium"):
        result["isbn13"] = best.get("isbn13", "")
        result["isbn10"] = best.get("isbn10", "")
        result["isbn_status"] = "Edition match" if confidence == "High" else "Probable edition match"
    elif confidence == "Low":
        result["isbn_status"] = "Candidate only: exact edition unconfirmed"
    elif likely_pre_isbn(book):
        result["isbn_status"] = "No safe ISBN assignment: edition likely predates ISBN"
    else:
        result["isbn_status"] = "No safe ISBN assignment"
    return result


def query_candidates(book: dict[str, Any]) -> list[dict[str, Any]]:
    google = google_query(book)
    best_google_title = max((similarity(book.get("title"), item.get("title")) for item in google), default=0.0)
    if google and best_google_title >= 0.72:
        return google
    return google + openlibrary_query(book)


def main() -> None:
    books = load_books()
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    for book in books:
        key = (norm(book.get("title")), norm(primary_author(book.get("author"))))
        unique.setdefault(key, book)
    print(f"Loaded {len(books)} books; querying {len(unique)} unique title/author combinations", flush=True)
    candidate_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        future_to_key = {executor.submit(query_candidates, book): key for key, book in unique.items()}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_key), start=1):
            key = future_to_key[future]
            try:
                candidate_cache[key] = future.result()
            except Exception as exc:
                print(f"Query failed for {key}: {exc}", flush=True)
                candidate_cache[key] = []
            if index % 50 == 0 or index == len(future_to_key):
                print(f"Queried {index}/{len(future_to_key)}", flush=True)
    results = []
    for book in books:
        key = (norm(book.get("title")), norm(primary_author(book.get("author"))))
        results.append(match_book(book, candidate_cache.get(key, [])))
    for chunk_index in range(0, len(results), 100):
        chunk = results[chunk_index:chunk_index + 100]
        (RESULTS_DIR / f"isbn_results_{chunk_index // 100:02d}.json").write_text(json.dumps(chunk, ensure_ascii=False, indent=2), encoding="utf-8")
    counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    for result in results:
        counts[result["isbn_status"]] = counts.get(result["isbn_status"], 0) + 1
        confidence_counts[result["isbn_confidence"]] = confidence_counts.get(result["isbn_confidence"], 0) + 1
    summary = {"total": len(results), "with_primary_isbn": sum(1 for row in results if row["isbn13"] or row["isbn10"]), "with_candidate_only": sum(1 for row in results if row["candidate_isbn13"] or row["candidate_isbn10"]), "status_counts": counts, "confidence_counts": confidence_counts}
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
