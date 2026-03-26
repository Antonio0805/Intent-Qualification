#!/usr/bin/env python3
"""
Intent Qualification System for Company Search
Pipeline: Query Analysis → Structured Filter → Embedding Rank → LLM Batch Qualify
"""

import ast
import json
import os
import re
import concurrent.futures
from typing import Optional

import numpy as np
from anthropic import Anthropic

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Skipping embedding stage.")

client = Anthropic()
HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"

QUERIES = [
    "Logistic companies in Romania",
    "Public software companies with more than 1,000 employees",
    "Food and beverage manufacturers in France",
    "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    "Construction companies in the United States with revenue over $50 million",
    "Pharmaceutical companies in Switzerland",
    "B2B SaaS companies providing HR solutions in Europe",
    "Clean energy startups founded after 2018 with fewer than 200 employees",
    "Fast-growing fintech companies competing with traditional banks in Europe",
    "E-commerce companies using Shopify or similar platforms",
    "Renewable energy equipment manufacturers in Scandinavia",
    "Companies that manufacture or supply critical components for electric vehicle battery production",
]

# Queries that need deeper reasoning get Sonnet, rest get Haiku
COMPLEX_QUERIES = {
    "Companies that could supply packaging materials for a direct-to-consumer cosmetics brand",
    "Fast-growing fintech companies competing with traditional banks in Europe",
    "Companies that manufacture or supply critical components for electric vehicle battery production",
}

COUNTRY_CODES = {
    "romania": "ro", "romanian": "ro",
    "germany": "de", "german": "de",
    "france": "fr", "french": "fr",
    "switzerland": "ch", "swiss": "ch",
    "united states": "us", "usa": "us", "america": "us", "american": "us",
    "united kingdom": "gb", "uk": "gb", "british": "gb",
    "sweden": "se", "swedish": "se",
    "norway": "no", "norwegian": "no",
    "denmark": "dk", "danish": "dk",
    "finland": "fi", "finnish": "fi",
    "netherlands": "nl", "dutch": "nl",
    "spain": "es", "spanish": "es",
    "italy": "it", "italian": "it",
    "china": "cn", "chinese": "cn",
    "india": "in", "indian": "in",
    "australia": "au", "australian": "au",
    "canada": "ca", "canadian": "ca",
}

REGION_GROUPS = {
    "scandinavia": ["se", "no", "dk", "fi"],
    "nordic": ["se", "no", "dk", "fi"],
    "europe": [
        "ro", "de", "fr", "ch", "gb", "es", "fi", "se", "no", "dk",
        "nl", "it", "be", "at", "pl", "cz", "hu", "pt", "ie", "gr",
        "hr", "sk", "si", "lt", "lv", "ee", "bg", "rs", "ua", "md",
    ],
}


# ── Data Loading ───────────────────────────────────────────────────────────────

def load_companies(path: str = "companies.jsonl") -> list[dict]:
    companies = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                companies.append(json.loads(line))
    return companies


def parse_address(addr) -> dict:
    if not addr:
        return {}
    if isinstance(addr, dict):
        return addr
    try:
        return ast.literal_eval(addr)
    except Exception:
        return {}


def parse_naics(naics) -> Optional[dict]:
    if not naics:
        return None
    if isinstance(naics, dict):
        return naics
    try:
        return ast.literal_eval(naics)
    except Exception:
        return None


def parse_secondary_naics(sec) -> list[dict]:
    if not sec:
        return []
    if isinstance(sec, list):
        return sec
    try:
        result = ast.literal_eval(sec)
        return result if isinstance(result, list) else []
    except Exception:
        return []


def company_to_text(company: dict) -> str:
    """Compact text representation of a company for embedding / LLM context."""
    parts = []

    if company.get("operational_name"):
        parts.append(f"Name: {company['operational_name']}")

    addr = parse_address(company.get("address"))
    if addr:
        loc = [x for x in [addr.get("town"), addr.get("region_name"),
                            (addr.get("country_code") or "").upper()] if x]
        if loc:
            parts.append(f"Location: {', '.join(loc)}")

    naics = parse_naics(company.get("primary_naics"))
    if naics:
        parts.append(f"Primary Industry: {naics.get('label', '')}")

    sec_naics = parse_secondary_naics(company.get("secondary_naics"))
    if sec_naics:
        labels = [n.get("label", "") for n in sec_naics if isinstance(n, dict)]
        if labels:
            parts.append(f"Secondary Industries: {', '.join(labels)}")

    if company.get("description"):
        parts.append(f"Description: {company['description']}")

    bm = company.get("business_model")
    if bm:
        if not isinstance(bm, list):
            bm = [bm]
        parts.append(f"Business Model: {', '.join(bm)}")

    tm = company.get("target_markets")
    if tm:
        if not isinstance(tm, list):
            tm = [tm]
        parts.append(f"Target Markets: {', '.join(tm)}")

    co = company.get("core_offerings")
    if co:
        if not isinstance(co, list):
            co = [co]
        parts.append(f"Core Offerings: {', '.join(co[:8])}")

    if company.get("employee_count") is not None:
        parts.append(f"Employees: {int(company['employee_count'])}")

    if company.get("revenue") is not None:
        parts.append(f"Revenue: ${company['revenue']:,.0f}")

    if company.get("year_founded") is not None:
        parts.append(f"Founded: {int(company['year_founded'])}")

    if company.get("is_public") is not None:
        parts.append(f"Public: {'Yes' if company['is_public'] else 'No'}")

    return "\n".join(parts)


# ── Stage 1: Query Analysis ────────────────────────────────────────────────────

def analyze_query(query: str) -> dict:
    """
    Ask LLM to extract structured constraints + semantic description from query.
    Returns a constraints dict.
    """
    prompt = f"""Analyze this company search query and extract structured constraints.

Query: "{query}"

Return a JSON object with these fields (null if not applicable):
{{
  "countries": [],          // ISO 2-letter codes, or region keys: "scandinavia", "europe"
  "is_public": null,        // true, false, or null
  "min_revenue": null,      // number in USD
  "max_revenue": null,
  "min_employees": null,    // number
  "max_employees": null,
  "min_year_founded": null, // year integer
  "max_year_founded": null,
  "semantic_description": ""  // 1-2 sentences describing what fully qualifies as a match
}}

Country codes: ro=Romania, de=Germany, fr=France, ch=Switzerland, us=USA, gb=UK,
se=Sweden, no=Norway, dk=Denmark, fi=Finland, nl=Netherlands, es=Spain.
For "Scandinavia" use ["se","no","dk","fi"]. For "Europe" use ["europe"].
Revenue "$50M" → 50000000. "1,000 employees" → min_employees: 1000.
"Founded after 2018" → min_year_founded: 2019.
"Fewer than 200 employees" → max_employees: 199.

Return ONLY valid JSON, no markdown fences."""

    response = client.messages.create(
        model=HAIKU,
        max_tokens=400,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {
            "countries": [], "is_public": None,
            "min_revenue": None, "max_revenue": None,
            "min_employees": None, "max_employees": None,
            "min_year_founded": None, "max_year_founded": None,
            "semantic_description": query,
        }


# ── Stage 2: Structured Filter ─────────────────────────────────────────────────

def expand_countries(countries: list) -> set:
    result = set()
    for c in countries:
        cl = c.lower()
        if cl in REGION_GROUPS:
            result.update(REGION_GROUPS[cl])
        elif cl in COUNTRY_CODES:
            result.add(COUNTRY_CODES[cl])
        else:
            result.add(cl)
    return result


def passes_filters(company: dict, c: dict) -> bool:
    # Country
    if c.get("countries"):
        allowed = expand_countries(c["countries"])
        addr = parse_address(company.get("address"))
        cc = (addr.get("country_code") or "").lower()
        if cc not in allowed:
            return False

    # is_public
    if c.get("is_public") is not None:
        if company.get("is_public") != c["is_public"]:
            return False

    # Revenue
    rev = company.get("revenue")
    if c.get("min_revenue") is not None and rev is not None:
        if rev < c["min_revenue"]:
            return False
    if c.get("max_revenue") is not None and rev is not None:
        if rev > c["max_revenue"]:
            return False

    # Employees
    emp = company.get("employee_count")
    if c.get("min_employees") is not None and emp is not None:
        if emp < c["min_employees"]:
            return False
    if c.get("max_employees") is not None and emp is not None:
        if emp > c["max_employees"]:
            return False

    # Year founded
    yr = company.get("year_founded")
    if c.get("min_year_founded") is not None and yr is not None:
        if yr < c["min_year_founded"]:
            return False
    if c.get("max_year_founded") is not None and yr is not None:
        if yr > c["max_year_founded"]:
            return False

    return True


def structured_filter(companies: list, constraints: dict) -> list:
    filtered = [c for c in companies if passes_filters(c, constraints)]
    # If filters removed everything (e.g. all employees unknown), fall back
    if not filtered:
        # Relax numeric-only constraints and retry with just country/public
        relaxed = {
            "countries": constraints.get("countries", []),
            "is_public": constraints.get("is_public"),
            "min_revenue": None, "max_revenue": None,
            "min_employees": None, "max_employees": None,
            "min_year_founded": None, "max_year_founded": None,
        }
        filtered = [c for c in companies if passes_filters(c, relaxed)]
    if not filtered:
        filtered = companies  # full fallback
    return filtered


# ── Stage 3: Embedding Ranking ─────────────────────────────────────────────────

_model = None
_company_embeddings: Optional[np.ndarray] = None
_company_texts: Optional[list] = None


def load_embedding_model():
    global _model
    if _model is None and EMBEDDINGS_AVAILABLE:
        print("Loading embedding model (all-MiniLM-L6-v2)...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def precompute_embeddings(companies: list) -> None:
    """Embed all companies once at startup."""
    global _company_embeddings, _company_texts
    model = load_embedding_model()
    if model is None:
        return
    print(f"Pre-computing embeddings for {len(companies)} companies...")
    _company_texts = [company_to_text(c) for c in companies]
    _company_embeddings = model.encode(
        _company_texts, normalize_embeddings=True,
        batch_size=64, show_progress_bar=True
    ).astype(np.float64)
    print("Embeddings ready.")


def embedding_rank(query: str, semantic_desc: str, company_indices: list,
                   top_k: int = 60) -> list:
    """
    Rank company_indices by cosine similarity with query.
    Returns list of (original_index, score) sorted descending, capped at top_k.
    """
    model = load_embedding_model()
    if model is None or _company_embeddings is None:
        return [(i, 0.5) for i in company_indices[:top_k]]

    q_text = semantic_desc if semantic_desc else query
    q_emb = model.encode(q_text, normalize_embeddings=True).astype(np.float64)

    indices = np.array(company_indices)
    embs = _company_embeddings[indices]
    scores = embs @ q_emb  # cosine sim (already normalized)

    order = np.argsort(-scores)
    top = order[:top_k]
    return [(int(company_indices[i]), float(scores[i])) for i in top]


# ── Stage 4: LLM Batch Qualification ──────────────────────────────────────────

def qualify_batch(query: str, batch: list[tuple], model_id: str) -> list[dict]:
    """
    batch: list of (company_dict, emb_score)
    Returns list of result dicts with match/score/reason.
    """
    if not batch:
        return []

    blocks = []
    for i, (company, _) in enumerate(batch):
        blocks.append(f"COMPANY {i+1} [{company.get('operational_name', 'Unknown')}]:\n{company_to_text(company)}")

    companies_text = "\n\n---\n\n".join(blocks)

    prompt = f"""Qualify each company for the following search query. Be strict about intent.

Query: "{query}"

Rules:
- A company must DO the thing queried, not just serve or supply that industry.
  Example: "logistics companies" → must provide logistics services, NOT logistics software companies.
  Example: "packaging suppliers for cosmetics" → must supply packaging, NOT cosmetics brands.
- Location must match if specified.
- Numeric constraints (revenue, employees, year) must be satisfied if data is available.
  If data is missing, give benefit of the doubt for numeric constraints only.

For each company return a JSON array:
[
  {{"id": 1, "name": "...", "match": true/false, "score": 1-5, "reason": "one sentence"}},
  ...
]
score: 5=perfect match, 4=strong match, 3=borderline, 2=weak, 1=no match

Companies:
{companies_text}

Return ONLY a valid JSON array. No markdown."""

    response = client.messages.create(
        model=model_id,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()
    text = re.sub(r"^```(?:json)?\n?", "", text)
    text = re.sub(r"\n?```$", "", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: mark all as borderline
        return [
            {"id": i + 1, "name": c[0].get("operational_name", ""),
             "match": True, "score": 3, "reason": "LLM parse error – kept as borderline"}
            for i, c in enumerate(batch)
        ]


def llm_qualify(query: str, ranked: list[tuple], companies: list,
                batch_size: int = 10, model_id: str = HAIKU) -> list[dict]:
    """
    ranked: list of (company_index, emb_score)
    Returns qualified results sorted by match then score.
    """
    batches = [ranked[i:i + batch_size] for i in range(0, len(ranked), batch_size)]
    # Prepare (company_dict, emb_score) batches
    company_batches = [
        [(companies[idx], score) for idx, score in b]
        for b in batches
    ]

    all_results = []

    def process_batch(args):
        b_idx, company_batch = args
        return b_idx, qualify_batch(query, company_batch, model_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_batch, (i, b)): i
                   for i, b in enumerate(company_batches)}
        batch_results = {}
        for future in concurrent.futures.as_completed(futures):
            b_idx, results = future.result()
            batch_results[b_idx] = results

    for b_idx in sorted(batch_results.keys()):
        results = batch_results[b_idx]
        batch = company_batches[b_idx]
        for j, res in enumerate(results):
            if j < len(batch):
                company, emb_score = batch[j]
                all_results.append({
                    "company": company,
                    "match": res.get("match", False),
                    "llm_score": res.get("score", 3),
                    "emb_score": emb_score,
                    "reason": res.get("reason", ""),
                })

    all_results.sort(key=lambda x: (-int(x["match"]), -x["llm_score"], -x["emb_score"]))
    return all_results


# ── Main Pipeline ──────────────────────────────────────────────────────────────

def qualify_query(query: str, companies: list, top_k: int = 60,
                  batch_size: int = 10, verbose: bool = True) -> list[dict]:
    model_id = SONNET if query in COMPLEX_QUERIES else HAIKU

    if verbose:
        print(f"\n{'='*60}")
        print(f"Query : {query}")
        print(f"Model : {model_id}")

    # Stage 1: analyze
    constraints = analyze_query(query)
    if verbose:
        print(f"Filter: countries={constraints.get('countries')}, "
              f"public={constraints.get('is_public')}, "
              f"emp=[{constraints.get('min_employees')},{constraints.get('max_employees')}], "
              f"rev=[{constraints.get('min_revenue')},{constraints.get('max_revenue')}], "
              f"yr=[{constraints.get('min_year_founded')},{constraints.get('max_year_founded')}]")

    # Stage 2: structured filter → get indices
    filtered_indices = [i for i, c in enumerate(companies) if passes_filters(c, constraints)]
    if not filtered_indices:
        # Relax numeric constraints, keep country/public
        relaxed = {
            "countries": constraints.get("countries", []),
            "is_public": constraints.get("is_public"),
            "min_revenue": None, "max_revenue": None,
            "min_employees": None, "max_employees": None,
            "min_year_founded": None, "max_year_founded": None,
        }
        filtered_indices = [i for i, c in enumerate(companies) if passes_filters(c, relaxed)]
    if not filtered_indices:
        filtered_indices = list(range(len(companies)))
    if verbose:
        print(f"After filter: {len(filtered_indices)}/{len(companies)} companies")

    # Stage 3: embedding rank
    semantic_desc = constraints.get("semantic_description", query)
    ranked = embedding_rank(query, semantic_desc, filtered_indices, top_k=min(top_k, len(filtered_indices)))
    if verbose:
        print(f"After embedding rank: top {len(ranked)} candidates")

    # Stage 4: LLM qualify
    results = llm_qualify(query, ranked, companies, batch_size=batch_size, model_id=model_id)
    if verbose:
        matches = sum(1 for r in results if r["match"])
        print(f"Qualified: {matches} matches out of {len(results)} evaluated")

    return results


def format_results(results: list, query: str, show_top: int = 20) -> str:
    lines = [f"\n{'='*60}", f"QUERY: {query}", "="*60]
    matches = [r for r in results if r["match"]]
    lines.append(f"Matched {len(matches)} companies\n")
    for i, r in enumerate(matches[:show_top], 1):
        c = r["company"]
        addr = parse_address(c.get("address"))
        loc = f"{addr.get('town','')}, {(addr.get('country_code') or '').upper()}".strip(", ")
        emp = f"{int(c['employee_count'])} emp" if c.get("employee_count") else ""
        rev = f"${c['revenue']/1e6:.0f}M rev" if c.get("revenue") else ""
        meta = " | ".join(filter(None, [loc, emp, rev]))
        name = c.get('operational_name') or '?'
        lines.append(f"  {i:2}. [{r['llm_score']}/5] {name:40s}  {meta}")
        lines.append(f"       {r['reason']}")
    return "\n".join(lines)


# ── Entry Point ────────────────────────────────────────────────────────────────

def main():
    data_path = "companies.jsonl"
    if not os.path.exists(data_path):
        data_path = os.path.join(os.path.dirname(__file__), "companies.jsonl")

    print("Loading companies...")
    companies_raw = load_companies(data_path)
    # Deduplicate by website, keeping the record with more data
    seen_websites: dict = {}
    for c in companies_raw:
        key = c.get("website") or c.get("operational_name") or id(c)
        if key not in seen_websites:
            seen_websites[key] = c
        else:
            # Keep the one with more non-null fields
            existing = seen_websites[key]
            if sum(v is not None for v in c.values()) > sum(v is not None for v in existing.values()):
                seen_websites[key] = c
    companies = list(seen_websites.values())
    print(f"Loaded {len(companies_raw)} companies, {len(companies)} after deduplication")

    if EMBEDDINGS_AVAILABLE:
        precompute_embeddings(companies)

    all_results: dict[str, list] = {}

    for query in QUERIES:
        results = qualify_query(query, companies, top_k=60, batch_size=10)
        all_results[query] = results
        print(format_results(results, query))

    # Save to JSON
    output = {}
    for query, results in all_results.items():
        output[query] = [
            {
                "company": r["company"].get("operational_name"),
                "website": r["company"].get("website"),
                "match": r["match"],
                "llm_score": r["llm_score"],
                "emb_score": round(r["emb_score"], 4),
                "reason": r["reason"],
            }
            for r in results
        ]
    with open("results.json", "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print("\nResults saved to results.json")


if __name__ == "__main__":
    main()
