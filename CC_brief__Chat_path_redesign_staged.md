# Claude Code brief: Chat path redesign — staged implementation

## Context

The current chat path embeds the user query, runs a Chroma similarity
search with a percentage threshold, then passes surviving results to
Haiku for gatekeeping. This architecture has two weaknesses:

1. Retrieval happens before reasoning — Haiku never sees results that
   fell below the threshold, even if they are genuinely relevant
2. Context augmentation biases retrieval toward the session history,
   causing domain swerves and category queries to fail

This brief redesigns the chat path so Haiku directs retrieval rather
than filtering its output. Haiku specifies what to search for before
any Chroma query fires. The backend executes those instructions
faithfully.

**Implement in four stages. Commit and test after each stage before
proceeding. A clean fallback point must exist at every stage.**

---

## Before starting

Create a branch for this work. `main` stays untouched and is the
fallback at any point.

```
git add -A
git commit -m "Pre chat-path-redesign snapshot"
git checkout -b chat-path-redesign
```

All staged commits go on `chat-path-redesign`. If anything goes wrong
at any stage, return to `main` instantly:

```
git checkout main
```

The branch is preserved — you can return to it, inspect it, or
abandon it without affecting `main`.

---

## Stage 1 — specify_searches tool, logging only

**Goal:** Get Haiku producing structured search specifications. Verify
the classifications look sensible. No change to retrieval or results.

### The specify_searches tool

Add this tool to the Haiku call on the chat path:

```json
{
  "name": "specify_searches",
  "description": "Specify what searches to run to answer the user query. Called before any retrieval happens.",
  "input_schema": {
    "type": "object",
    "properties": {
      "query_type": {
        "type": "string",
        "enum": ["filter", "intent", "refine", "swerve", "out_of_scope"],
        "description": "filter: structured category/field request. intent: interest or goal expression. refine: narrowing current candidate set. swerve: domain change mid-session. out_of_scope: unrelated to courses or careers."
      },
      "searches": {
        "type": "array",
        "description": "One or more searches to run. Use multiple searches to cover different angles on the same intent.",
        "items": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "Expanded search query — rephrase the user's words into terminology likely to appear in course and job descriptions"
            },
            "type": {
              "type": "string",
              "enum": ["courses", "jobs", "both"]
            },
            "scope": {
              "type": "string",
              "enum": ["candidate_set", "full_collection"],
              "description": "candidate_set: search within active candidate set only. full_collection: search entire Chroma collection."
            },
            "filters": {
              "type": "object",
              "description": "Optional structured field filters to apply alongside semantic search",
              "properties": {
                "ssa_label": {"type": "string"},
                "qual_type": {"type": "array", "items": {"type": "string"}},
                "mode": {"type": "string", "enum": ["FT", "PT", "FT/PT"]},
                "provider": {"type": "string"},
                "level": {"type": "integer"}
              }
            }
          },
          "required": ["query", "type", "scope"]
        }
      },
      "collection_action": {
        "type": "string",
        "enum": ["build", "refine", "replace", "none"],
        "description": "build: create new candidate set from results. refine: narrow existing set. replace: discard existing set and build new one. none: return focal card(s) only."
      },
      "acknowledgement": {
        "type": "string",
        "description": "One sentence for the bottom zone — what you understood and are doing"
      }
    },
    "required": ["query_type", "searches", "collection_action", "acknowledgement"]
  }
}
```

Force tool use: `tool_choice = {"type": "tool", "name": "specify_searches"}`

### Updated system prompt for Haiku

Replace the existing chat path system prompt with:

```
You are a search director for a course and career exploration app.
When a user sends a query, you specify what searches to run before
any results are fetched.

Think carefully about what the user is asking:
- "filter" queries name a category, subject, or field constraint
  e.g. "show me construction courses", "part time health jobs"
- "intent" queries express an interest, goal, or personal situation
  e.g. "I want to work outdoors", "I'm good with my hands"
- "refine" queries narrow what the user is already looking at
  e.g. "just the HND ones", "which of these are near Wigan"
- "swerve" queries change domain mid-session
  e.g. "actually show me digital courses instead"
- "out_of_scope" queries are unrelated to courses or careers

For search queries, expand the user's words into terminology likely
to appear in course and job descriptions. "NHS" becomes "healthcare
clinical nursing hospital medical". "Building industry" becomes
"construction civil engineering built environment site management".

You may specify multiple searches to cover different angles on the
same intent. Each search can target courses, jobs, or both.

If a candidate set is active, decide whether to search within it
(refine) or the full collection (new search).

The app has these subject areas: Engineering and Manufacturing
Technologies, Information and Communication Technology, Construction
Planning and the Built Environment, Health Public Services and Care,
Arts Media and Publishing, Business Administration and Law.

Available providers: Wigan & Leigh College, University of Salford,
Trafford & Stockport College, Tameside College, Bury College,
Ada College.
```

### Stage 1 implementation

In the chat endpoint in api.py:

1. Call Haiku with the new tool and system prompt
2. Extract the `specify_searches` tool result
3. **Log the full specification to console** — do not change retrieval
4. Continue with the existing retrieval path as before
5. Return results as now

Example log output:
```
[SPECIFY_SEARCHES]
  query_type: filter
  searches:
    - query: "healthcare clinical nursing care physiotherapy"
      type: courses
      scope: full_collection
      filters: {ssa_label: "Health, Public Services and Care"}
  collection_action: build
  acknowledgement: "Searching for health and care courses across GM IoT"
```

### Stage 1 acceptance checks

- [ ] Haiku produces valid `specify_searches` tool output for every query
- [ ] `query_type` classifications look correct for a range of queries:
  - "show me construction courses" → filter
  - "I want to work with people" → intent
  - "just the part time ones" → refine
  - "actually show me digital instead" → swerve
  - "what is the weather like" → out_of_scope
- [ ] Search queries show meaningful expansion:
  - "NHS" → includes healthcare/clinical/nursing terminology
  - "building industry" → includes construction/civil/built environment
- [ ] Existing results unchanged — users see no difference yet
- [ ] No errors on any query type

### Stage 1 git commit

```
git add -A
git commit -m "Stage 1: specify_searches tool — logging only

- Haiku now produces structured search specification before retrieval
- query_type classification, expanded queries, field filters
- Specification logged to console — retrieval path unchanged
- No change to results or user experience
- Branch: chat-path-redesign"
```

---

## Stage 2 — backend executes specified searches

**Goal:** Replace the existing retrieval path with execution of Haiku's
specified searches. Top-N results, no percentage threshold.

### Top-N configuration

```python
SEARCH_TOP_N = 8  # results per search
```

No percentage threshold. Haiku will filter irrelevant results in
Stage 4. Top-N is a relative quality judgement — the best available
matches regardless of absolute score.

### Search execution

For each search in `specify_searches.searches`:

1. Embed `search.query` using Voyage AI with `input_type="query"`
2. Build Chroma query parameters:
   - `n_results = SEARCH_TOP_N`
   - `where` clause from `search.filters` if present
   - Collection: `gmiot_courses` for courses, `gmiot_jobs` for jobs
3. Run the query
4. Collect results with metadata

If multiple searches specified, merge results and deduplicate by ID
before returning to Haiku.

### Candidate set scoping

If `search.scope == "candidate_set"` and a candidate set is active
in the session, add an ID filter to the Chroma query:

```python
where = {"course_id": {"$in": candidate_course_ids}}
```

Merge with any field filters from `search.filters` using `$and`.

If no candidate set is active, fall back to `full_collection`
regardless of scope specified.

### Chroma filter construction

Build the `where` clause from `search.filters`:

```python
def build_where_clause(filters, id_scope=None):
    conditions = []
    if id_scope:
        conditions.append({"course_id": {"$in": id_scope}})
    if filters.get("ssa_label"):
        conditions.append({"ssa_label": {"$eq": filters["ssa_label"]}})
    if filters.get("qual_type"):
        conditions.append({"qualification_type": {"$in": filters["qual_type"]}})
    if filters.get("mode"):
        conditions.append({"mode": {"$eq": filters["mode"]}})
    if filters.get("provider"):
        conditions.append({"provider": {"$eq": filters["provider"]}})
    if filters.get("level"):
        conditions.append({"level": {"$eq": filters["level"]}})
    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}
```

### Stage 2 implementation

Replace the existing Chroma retrieval in the chat endpoint with the
new execution loop. The Haiku gatekeeping call (current second Haiku
call) is temporarily bypassed — return all results directly to the
frontend.

This is a temporary state to verify retrieval quality before adding
Stage 4. Results will be unfiltered by Haiku — more results may show,
some may be less relevant. This is expected and temporary.

### Stage 2 acceptance checks

- [ ] "I want to work in the health sector" returns health courses
  and jobs (previously failing)
- [ ] "Show me construction courses" returns construction courses
- [ ] "Near Wigan" applies provider filter to Wigan & Leigh College
- [ ] "Part time" applies mode filter correctly
- [ ] Multiple searches merge and deduplicate correctly
- [ ] Candidate set scoping narrows results when a set is active
- [ ] No percentage threshold errors — all top-N results returned

### Stage 2 git commit

```
git add -A
git commit -m "Stage 2: backend executes specified searches

- Chroma retrieval now driven by Haiku's specify_searches output
- Top-N replaces percentage threshold
- Field filters applied from Haiku specification
- Candidate set scoping implemented
- Haiku gatekeeping temporarily bypassed — all results returned
- Branch: chat-path-redesign"
```

---

## Stage 3 — candidate set creation and management

**Goal:** Materialise candidate sets from search results. Tile-built
sets from existing tile path also feed into the same mechanism.

### Candidate set data structure

Store in the session (server-side, keyed by session ID or passed
from frontend):

```python
candidate_set = {
    "course_ids": [1, 4, 7, 12, 23],   # list of course_id integers
    "job_ids": [5, 18, 42],              # list of job id integers
    "built_from": "Construction · HND",  # description for display
    "collection_action": "build"
}
```

### collection_action handling

- `build` — create new candidate set from search results
- `refine` — intersect new results with existing candidate set
- `replace` — discard existing set, build new one from results
- `none` — do not update candidate set, return results as focal cards

### Tile path integration

When a subject tile tap (and optional qual filter) returns results,
materialise a candidate set automatically:

```python
# After tile search returns results
candidate_set = {
    "course_ids": [r["course_id"] for r in results],
    "built_from": f"{subject_label}" + (f" · {qual_label}" if qual_filter else ""),
    "collection_action": "build"
}
```

Pass the candidate set summary to Haiku in subsequent chat calls:

```
Active candidate set: 21 Construction courses
(built from: Construction tile)
```

### Stage 3 implementation

- Add candidate set storage to session state
- Update tile path to materialise candidate set on results
- Pass candidate set summary to Haiku in the system context
- Handle all four `collection_action` values
- Do not yet update the frontend list card — the candidate set
  lives server-side for now

### Stage 3 acceptance checks

- [ ] Tile tap creates candidate set with correct course IDs
- [ ] Qual filter refines the candidate set correctly
- [ ] Chat query with `scope: candidate_set` searches within set
- [ ] `collection_action: refine` narrows existing set
- [ ] `collection_action: replace` discards and rebuilds
- [ ] `collection_action: none` returns results without updating set
- [ ] Candidate set summary passed to Haiku in context

### Stage 3 git commit

```
git add -A
git commit -m "Stage 3: candidate set creation and management

- Tile tap materialises candidate set automatically
- Chat path updates candidate set per collection_action
- Candidate set scoping applied in Stage 2 retrieval
- Candidate set summary passed to Haiku as context
- Branch: chat-path-redesign"
```

---

## Stage 4 — two-turn Haiku pattern

**Goal:** Restore Haiku's selection and acknowledgement role. Haiku
now both specifies searches (turn 1) and selects from results (turn 2)
in a single two-turn conversation.

### Two-turn conversation structure

Turn 1 — Haiku specifies searches (existing after Stage 1):
```python
messages = [
    {"role": "user", "content": user_query_with_context}
]
response_1 = haiku_call(messages, tools=[specify_searches_tool])
spec = extract_tool_result(response_1)
```

Turn 2 — backend fetches results, passes back to Haiku:
```python
results_summary = format_results_for_haiku(retrieved_results)

messages = [
    {"role": "user", "content": user_query_with_context},
    {"role": "assistant", "content": response_1.content},  # Haiku's turn 1
    {"role": "user", "content": f"Here are the search results:\n{results_summary}\n\nSelect the most relevant results and provide your acknowledgement."}
]
response_2 = haiku_call(messages, tools=[select_results_tool])
selection = extract_tool_result(response_2)
```

### select_results_tool

```json
{
  "name": "select_results",
  "description": "Select the most relevant results from the search results provided",
  "input_schema": {
    "type": "object",
    "properties": {
      "approved_ids": {
        "type": "array",
        "items": {"type": "string"},
        "description": "IDs of results to show the user"
      },
      "rejected_ids": {
        "type": "array",
        "items": {"type": "string"},
        "description": "IDs of results to discard as irrelevant"
      },
      "acknowledgement": {
        "type": "string",
        "description": "One sentence for the bottom zone"
      },
      "advisory_trigger": {
        "type": "boolean",
        "description": "Whether to trigger an advisory card"
      }
    },
    "required": ["approved_ids", "acknowledgement", "advisory_trigger"]
  }
}
```

### Results summary format for Haiku

Keep it concise — Haiku needs enough to make a selection decision:

```
Courses found (8):
[1] Access to HE — Social Science · Bury College · Level 3
[4] HNC Health and Social Care · Wigan & Leigh · Level 4
[7] T Level — Health · Tameside · Level 3
...

Jobs found (8):
[101] Healthcare Assistant · NCS · £18k–£28k
[205] Physiotherapist · Prospects · £25k–£50k
...

User query: "I want to work in the health sector"
Candidate set active: No
```

### Latency consideration

Two Haiku calls adds latency. Haiku is fast (~200-400ms per call)
so total addition is modest. However if latency is noticeable in
testing, consider:

- Running turn 1 immediately on user input (speculative pre-fetch)
- Caching the specification for repeat queries
- Returning the acknowledgement from turn 1 to the bottom zone
  immediately while retrieval and turn 2 complete in background

Do not optimise prematurely — test latency first.

### Stage 4 acceptance checks

- [ ] Two-turn conversation completes without errors
- [ ] Haiku correctly rejects irrelevant results in turn 2
- [ ] Acknowledgement appears in bottom zone as before
- [ ] Advisory card triggering still works
- [ ] "I want to work in the health sector" returns relevant results
  and Haiku's acknowledgement reflects the intent correctly
- [ ] Domain swerve mid-session works — Haiku detects it and
  `collection_action: replace` starts fresh
- [ ] Latency acceptable — test a range of queries and note
  any that feel slow

### Stage 4 git commit

```
git add -A
git commit -m "Stage 4: two-turn Haiku pattern complete

- Haiku turn 1: specify_searches — directs retrieval
- Backend: executes specified searches, top-N
- Haiku turn 2: select_results — filters and acknowledges
- Percentage threshold removed from chat path
- Full chat path redesign complete
- Branch: chat-path-redesign"
```

---

## Fallback plan

At any stage, if results are worse than the pre-redesign baseline,
return to `main` instantly:

```
git checkout main
```

The `chat-path-redesign` branch is preserved — you can inspect it,
continue it later, or abandon it. Nothing on `main` is affected.

When all four stages are complete and verified, merge back to `main`:

```
git checkout main
git merge chat-path-redesign
git branch -d chat-path-redesign
```

---

## Do not change

- Tile path SQL retrieval — unchanged throughout
- Advisory card Haiku call — unchanged throughout
- Detail view endpoints — unchanged
- Save/pin mechanic — unchanged
- List card rendering threshold — unchanged
- Voyage AI embedding model and dimensions — unchanged

---

*PathwayIQ · March 2026 · Scott Clark Consultants*
