# Intent Qualification System — Writeup

## 3.1 Approach

When I first read this problem, my instinct was to just send everything to an LLM and call it a day. But that felt like cheating — and more importantly, it wouldn't scale at all. So I spent a while thinking about where the real cost and latency comes from, and how I could avoid it without sacrificing accuracy.

The answer I landed on is a **four-stage pipeline** that progressively narrows down candidates:

```
User Query
    │
    ▼
Stage 1: Query Analysis         (1 LLM call — parse intent into structured constraints)
    │
    ▼
Stage 2: Structured Filter      (pure Python — zero API cost)
    │
    ▼
Stage 3: Embedding Ranking      (local model, pre-computed — fast cosine similarity)
    │
    ▼
Stage 4: LLM Batch Qualify      (~6 parallel API calls — final decision)
    │
    ▼
Ranked results
```

**Stage 1 — Query Analysis**

I use Claude Haiku to parse the raw query into a structured JSON object with fields like `countries`, `is_public`, `min_revenue`, `max_employees`, `min_year_founded`, etc. This one call gives me two things: hard filters I can apply programmatically, and a clean semantic description of what a qualifying company actually looks like.

I deliberately separated "parse the query" from "evaluate the companies" — these are two different cognitive tasks and shouldn't be mixed.

**Stage 2 — Structured Pre-Filter**

Before touching any ML, I apply simple Python rules. If the query says "in Romania", I filter to `country_code == 'ro'`. If it says "public companies", I check `is_public`. Revenue, employee count, founding year — all handled as direct comparisons.

This step is free (no API) and can eliminate 80-90% of the dataset instantly for geographic or numeric queries. It's boring but effective.

I also handle regions: "Scandinavia" expands to `[se, no, dk, fi]`, "Europe" to a full list of country codes.

One edge case: if all companies get filtered out (usually because employee_count is null for most companies), I relax the numeric constraints and retry with just the geographic/boolean filters, then fall back to the full dataset if needed.

**Stage 3 — Embedding Similarity**

I use `sentence-transformers` with `all-MiniLM-L6-v2` — small, fast, runs locally, no API cost. I pre-compute embeddings for all 477 companies once at startup and cache them as a NumPy matrix. For each query, I encode the semantic description (not the raw query — the cleaner version from Stage 1) and do a matrix multiply to get cosine similarities in milliseconds.

I take the top 60 from the filtered set. This is where I'd expect to lose some edge cases, but it keeps Stage 4 cheap.

**Stage 4 — LLM Batch Qualification**

I send the top 60 candidates to the LLM in batches of 10, running all batches in parallel (ThreadPoolExecutor). Each batch call returns a `match` (true/false), a `score` (1–5), and a one-sentence reason per company.

The prompt is deliberately strict: "a company must DO the thing queried, not just serve that industry." This is the key fix for the embedding failure mode — embeddings would rank cosmetics brands highly for "packaging suppliers for cosmetics brands", but the LLM understands the difference.

I use Claude Haiku for simple/structured queries and Claude Sonnet for the three queries that require genuine reasoning (packaging suppliers, fintech vs banks, EV battery components). This saves cost without sacrificing quality where it matters.

**Total cost per query: ~7 LLM calls** (1 analysis + ~6 qualification batches), vs. 477 for the naive baseline.

---

## 3.2 Tradeoffs

The main tension here is between accuracy and cost. I optimized for accuracy first, then tried to make it cheap enough to be practical.

A few specific tradeoffs I made consciously:

**Top-60 embedding cutoff.** I'm probably missing some true positives ranked 61+. I chose 60 because it felt like a reasonable budget for the LLM stage — at batch size 10, that's 6 calls. For higher recall, I'd bump this to 100 and accept the extra cost.

**Parallel LLM calls.** All batches within a query run in parallel. This cuts wall-clock time by ~4x but increases the risk of hitting rate limits. For a production system I'd add a semaphore — for this task it was fine.

**Missing data gets benefit of the doubt.** If a company doesn't have `employee_count` and the query says "fewer than 200 employees", I keep it in the filtered set rather than dropping it. This increases false positives slightly but avoids silently missing valid companies. The LLM stage can still reject them based on description signals.

**Haiku vs Sonnet routing.** I hardcoded which queries get Sonnet. A smarter system would detect query complexity dynamically — but for 12 known queries, explicit routing is simpler and more predictable.

---

## 3.3 Error Analysis

### Where it works well

For structured queries ("public software companies with more than 1,000 employees", "pharmaceutical companies in Switzerland"), the pipeline is almost perfect. The structured filter does the heavy lifting and the LLM just confirms.

For supply-chain queries with clear packaging/manufacturing signals in the data, Sonnet does a good job distinguishing suppliers from buyers.

### Where it struggles

**"E-commerce companies using Shopify or similar platforms"**

This is the query where I'm least confident. The dataset has no technology stack field — no company mentions "Shopify" in its description. The system ends up qualifying general e-commerce companies (Lululemon, Toys R Us) even though they almost certainly don't use Shopify. It's doing the best it can with the available data, but it's fundamentally an impossible constraint to verify from company profiles alone.

**"Fast-growing fintech companies competing with traditional banks in Europe"**

"Fast-growing" can't be derived from static data — there's no revenue CAGR or growth rate field. The system found 4 matches (Phin, Folio, Aircash, Agility FX), which is probably undercounting. Many fintech companies in the dataset might qualify but don't have descriptions that signal "competing with banks" explicitly enough.

**Duplicate companies**

Some companies appear twice in the results (e.g., World Wide Wind shows up twice for the clean energy query). This is a data quality issue — there are duplicate entries in the JSONL file with different records. My pipeline doesn't deduplicate because I didn't want to assume which record is more authoritative.

**Borderline NAICS mismatches**

A company classified as "Computer Systems Design" might primarily build logistics software — it would pass a "logistics companies" filter if I relied on NAICS, but the LLM correctly rejects it. This only works because I don't do NAICS-based filtering (I intentionally left it out after realizing NAICS codes in the data are often too broad).

---

## 3.4 Scaling

Right now, with 477 companies and a pre-computed NumPy matrix, everything fits in memory and the matmul takes milliseconds. At 100,000 companies, this breaks in a few ways:

**Memory:** 100K × 384-dim float32 embeddings = ~150MB. Still fine actually. At 10M companies it becomes a problem.

**ANN search:** Exact cosine similarity over 100K vectors is still fast enough (~50ms), but at scale I'd switch to approximate nearest neighbor search — something like Faiss, Qdrant, or Weaviate. You lose a few percent recall but gain orders of magnitude in speed.

**Structured filtering with a database:** Right now it's a Python loop over a list. For 100K+ companies, I'd move the data into PostgreSQL or Elasticsearch and let the database handle country/revenue/employee filters as indexed queries. Then embedding search only happens on the filtered subset.

**Pre-computing embeddings:** At 100K companies, re-embedding everything on startup isn't realistic. I'd run this offline as a batch job whenever company data changes, and store embeddings in the vector DB alongside the company records.

**LLM stage stays the same:** We're still only sending 60 candidates to the LLM per query. This part scales fine as long as the earlier stages do their job.

---

## 3.5 Failure Modes

**Confident wrong answers from misleading descriptions**

If a company's description is vague marketing copy ("we transform the logistics industry through innovation"), the LLM might classify it as a logistics company even if it's actually a SaaS company. The structured filter won't catch this and neither will embeddings.

**Geographic edge cases**

A company might be headquartered in Poland but operate primarily in Germany, or be a Romanian subsidiary of a German parent. My filter uses the `country_code` from the address field, which can mismatch the actual operational geography. I'd expect a few false negatives here.

**Embedding phase missing the right candidates**

If the query requires reading between the lines ("companies that could supply packaging for cosmetics"), the top-60 by embedding similarity might not include all true positives. For the cosmetics packaging query, embeddings will naturally surface cosmetics brands (high token overlap) over packaging suppliers — the LLM prompt fixes this for the 60 that make it through, but the real risk is a great packaging supplier ranked 61st that never gets evaluated.

**What I'd monitor in production**

- Match rate per query over time — sudden drops suggest the embedding cutoff is too aggressive
- LLM score distribution — if I'm seeing lots of 3s (borderline), the prompt or cutoff needs tuning
- Parse error rate on LLM responses — indicates model reliability issues
- Latency per stage — helps identify bottlenecks as data grows
- Manual spot-checks on known-good companies — are they appearing in the top-60?
