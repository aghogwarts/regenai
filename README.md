# regenai

# 1. Clone/copy the regenai folder to wherever you want

# 2. Create venv and install (only pathspec + rich needed for Module 1)

cd regenai
uv venv
uv pip install pathspec rich

# 3. Run the test

uv run python -m pytest tests/test_registry.py

# OR simply:

PYTHONPATH=src uv run python tests/test_registry.py

# 4. Try the CLI against your actual messy directory

uv pip install -e .
regenai --dry-run

# It will prompt for your directory path and show the registry

<hr>

This is a solid problem space. Let me break down the pipeline end-to-end and challenge some assumptions along the way, especially around GMMs.

## The Pipeline at a Glance

**Ingest → Parse → Chunk → Embed → Reduce → Cluster → Label → Summarize → Output MD**

Let's go layer by layer.

---

## 1. File Discovery & Parsing

The first decision that affects everything downstream: **what's a "document unit"?**

A directory might have `.py`, `.md`, `.txt`, `.json`, `.yaml`, `.docx`, `.pdf`, config files, READMEs, etc. Each needs a different parser, but more importantly — **file metadata is itself a clustering signal**. Two files in `backend/auth/` are more likely to be related than one in `backend/auth/` and one in `docs/marketing/`. Don't discard the directory tree structure; encode it as a feature.

Suggested approach: build a file registry first — walk the tree, classify each file by type, and store metadata (path, extension, size, last modified) before parsing any content.

---

## 2. Chunking — This Is Where Most Pipelines Silently Fail

You need **different chunking strategies for different content types**:

- **Prose/docs**: Semantic chunking at paragraph or section boundaries (not fixed token windows). Libraries like `langchain`'s `RecursiveCharacterTextSplitter` are a baseline, but section-header-aware splitting is better.
- **Code**: AST-based chunking — split at function/class/module boundaries. A 500-token sliding window over Python code will split a function in half and produce garbage embeddings. `tree-sitter` gives you language-agnostic ASTs.
- **Config/YAML/JSON**: These are often small enough to embed whole. If not, split by top-level keys.

Each chunk must carry its **provenance metadata**: source file path, chunk index, file type, directory depth. This metadata feeds into clustering later.

---

## 3. Embedding

Two considerations here:

**Model choice**: Code and prose live in different semantic spaces. A model like `voyage-code-3` or `nomic-embed-text` handles both reasonably, but if your directory is heavily mixed, consider whether a single embedding model captures cross-type relationships well enough. For a first pass, a single good model is fine — don't over-engineer this yet.

**Granularity question**: Do you embed chunks, or do you also create file-level embeddings (e.g., mean-pool all chunk embeddings for a file)? You'll want both. Chunk-level for fine-grained clustering, file-level for coarse project assignment.

---

## 4. The GMM Question — Let's Stress-Test This

GMMs give you what you want on paper: soft cluster assignments = confidence scores. A chunk belonging 70% to cluster A and 25% to cluster B maps directly to your "associated confidence score" idea. But there are real problems:

**Problem 1 — Dimensionality.** Embedding vectors are typically 768–1536 dimensions. GMMs estimate full covariance matrices, which means you need _far_ more data points than dimensions, or the covariance matrices become singular. You'd need dimensionality reduction first (UMAP to ~15–50 dims), but UMAP's output doesn't preserve Gaussian structure — it preserves topological relationships. So you're fitting Gaussians to non-Gaussian manifolds.

**Problem 2 — Choosing K.** GMMs require you to specify the number of components. You don't know how many "projects" are in the directory. BIC/AIC can help with model selection, but they're noisy in high dimensions and you'd need to fit multiple models.

**Problem 3 — Cluster shape assumption.** GMMs assume ellipsoidal clusters. Document embeddings in reduced space often form irregular, varying-density clusters — exactly what Gaussians struggle with.

**Where GMMs _do_ shine**: If you've already done a first-pass hard clustering and want to refine with soft assignments, or if your projects are well-separated and roughly equal-sized. In practice, that's rarely the case with organic directory structures.

### Alternatives Worth Considering

**HDBSCAN** — probably the strongest candidate. It's density-based, doesn't need K, handles variable-density clusters, has a built-in noise/outlier concept (important — not every file belongs to a project), and provides `probabilities_` for soft membership. The main downside: it doesn't natively do multi-membership (a chunk can't belong to two clusters). But you can use its `soft_clustering` mode via `all_points_membership_vectors`.

**BERTopic** — this is essentially the entire pipeline you're describing already packaged: embeddings → UMAP → HDBSCAN → c-TF-IDF for cluster labeling. It's battle-tested for exactly this kind of topical clustering. The risk is that it's opinionated and harder to customize, but it could be a strong baseline to benchmark against.

**A hybrid approach** (my recommendation):

1. HDBSCAN for initial hard clustering + noise detection
2. Then fit a GMM _on the discovered clusters_ (using HDBSCAN's output as initialization) to get soft assignments
3. This gives you the best of both: robust cluster discovery + probabilistic confidence scores

---

## 5. Structural Signal Injection — The Accuracy Lever

You said accuracy is the main focus. Here's the thing: **pure semantic similarity from embeddings isn't enough**. Two files might discuss similar concepts (e.g., "authentication") but belong to completely different projects. The structural signals that disambiguate are:

- **Directory co-location**: Files in the same folder/subfolder get an affinity boost
- **Import/reference graphs**: If `file_a.py` imports from `file_b.py`, they're related. Parse import statements, `#include`, `require()`, etc., and build an adjacency graph
- **Naming conventions**: Files named `test_auth.py` and `auth_service.py` are related
- **Temporal proximity**: Files last modified around the same time may belong to the same work effort

The question is how to incorporate these. Two options:

**Option A — Feature concatenation**: Add structural features to the embedding vector before clustering. Simple but mixing semantic and structural features in one space is messy.

**Option B — Multi-view clustering / graph-based refinement**: Cluster on embeddings first, then use structural signals to merge/split clusters. For example, if two semantically close chunks are in completely different directory subtrees with zero cross-references, maybe they shouldn't be in the same cluster. This is cleaner and more interpretable.

I'd go with Option B.

---

## 6. Cluster Labeling & Validation

Once you have clusters, you need to name them ("this is the auth system", "this is the data pipeline"). Two approaches:

- **c-TF-IDF** (what BERTopic uses): Extract distinctive terms per cluster. Fast, no LLM needed, but labels can be cryptic.
- **LLM-based labeling**: Send representative chunks from each cluster to an LLM and ask "what project/topic does this group of documents belong to?" More expensive, much better labels.

For validation, consider a **second-pass LLM check**: for borderline chunks (confidence < some threshold), ask the LLM "does this chunk belong more to Project A or Project B?" This is where you spend LLM budget on accuracy rather than blanketing everything.

---

## 7. Summarization Layer

This is the final stage and it's relatively straightforward once clustering is solid. The key architectural decision is **hierarchical vs. flat summarization**:

- **Flat**: Dump all chunks in a cluster into the LLM context and say "summarize." Works for small clusters, fails when total content exceeds context window.
- **Hierarchical** (recommended): Summarize individual chunks → summarize chunk summaries by sub-topic → produce final project summary. Map-reduce pattern essentially.

For code specifically, the summary should capture: what the code does, key APIs/interfaces, dependencies, and architectural patterns — not line-by-line descriptions.

---

## Proposed Architecture

```
Directory
  │
  ├─ File Registry (walk tree, classify, store metadata)
  │
  ├─ Parse Layer (type-specific parsers)
  │     ├─ tree-sitter for code
  │     ├─ markdown parser for .md
  │     └─ text extraction for docs/pdf
  │
  ├─ Chunk Layer (type-aware chunking + provenance metadata)
  │
  ├─ Embed Layer (single model, chunk-level + file-level pooled)
  │
  ├─ Reduce (UMAP → ~20-30 dims)
  │
  ├─ Cluster Layer
  │     ├─ HDBSCAN (hard clusters + noise detection)
  │     └─ GMM refinement (soft assignments / confidence scores)
  │
  ├─ Structural Refinement
  │     ├─ Import graph analysis
  │     ├─ Directory affinity
  │     └─ Merge/split post-processing
  │
  ├─ Label + Validate (LLM-based cluster naming, borderline review)
  │
  └─ Summarize (hierarchical map-reduce per cluster → output .md)
```

---

## Key Decision Points to Settle

1. **Chunk-level vs file-level clustering?** I'd say chunk-level with file-level aggregation — a file's project membership = majority cluster of its chunks, weighted by confidence.

2. **GMM vs HDBSCAN vs hybrid?** Start with HDBSCAN as baseline, measure cluster quality (silhouette score, manual spot checks), then try the hybrid. Don't start with GMM alone.

3. **Which embedding model?** Depends on whether code-heavy or doc-heavy. For mixed: `nomic-embed-text` or `voyage-code-3`. Evaluate on a small labeled subset.

4. **How much LLM budget?** The more you push accuracy to the clustering layer (algorithmic), the less you need the LLM to compensate. LLM should be for labeling, borderline resolution, and summarization — not for fixing bad clusters.

5. **Vector DB choice?** For this use case (batch processing, not real-time retrieval), you might not even need a full vector DB. FAISS or even in-memory numpy arrays with HDBSCAN directly could suffice. ChromaDB if you want persistence without infrastructure overhead.

What's your directory composition looking like — mostly code, mostly docs, or a chaotic mix? And are you thinking local-only (Ollama/local embeddings) or API-based (OpenAI/Voyage)?

<hr>

Let me read the PDF more carefully to make sure I have the full picture.Good, I have everything. Let me lay out a structured comparison across the three approaches — my brainstorm, the deep-research report, and the RAPTOR-based plan.

---

## Three Approaches Side by Side

### 1. Philosophical Framing

Each approach frames the problem differently, and that drives everything else.

**Deep Research Report** treats this as a **standard RAG pipeline with clustering bolted on**. It's linear: crawl → chunk → embed → cluster → summarize → validate. The clustering is a one-shot grouping step, not the architectural centerpiece. It reads like a literature survey stitched into a pipeline — solid references but it doesn't commit hard to any single design philosophy.

**RAPTOR PDF** treats this as a **recursive tree-building problem**. The core insight it borrows from the RAPTOR paper is that summaries themselves become new data points that get re-embedded and re-clustered, building a hierarchy bottom-up until you reach a single root node. The directory structure is then _derived_ from this tree by flipping it. This is the most opinionated of the three — it's essentially saying "the output structure should be emergent, not imposed."

**My brainstorm** treated this as a **multi-signal clustering problem with accuracy as the north star**. The emphasis was on not relying solely on semantic embeddings — injecting structural signals (import graphs, directory co-location, temporal proximity) and using a hybrid clustering approach (HDBSCAN for discovery, GMM for refinement). It's less opinionated about the output format and more about getting the groupings right.

---

### 2. Parsing & Ingestion

| Dimension        | Deep Research                                                         | RAPTOR PDF                                           | My Brainstorm                                                     |
| ---------------- | --------------------------------------------------------------------- | ---------------------------------------------------- | ----------------------------------------------------------------- |
| Parser           | Apache Tika (heavy, Java-based, broad format support)                 | Unstructured.io (Python-native, modern, local-first) | Type-specific: tree-sitter for code, dedicated parsers per format |
| Metadata capture | Mentioned (timestamps, author, format) but not deeply used downstream | Minimal — treats files as text blobs                 | Central — metadata is a clustering feature, not just decoration   |

**The key disagreement**: The RAPTOR plan and deep research report both funnel everything into "text chunks" early and lose structural information. My brainstorm argues that's premature — a `.py` file and a `.md` file discussing the same topic should be chunked differently, and their _relationship_ (e.g., one imports the other) is a signal you can't recover after flattening to text.

**My take now**: Unstructured.io (RAPTOR's choice) is the better default parser — it's Python-native, actively maintained, handles most formats, and runs locally. But **code files should bypass it entirely** and go through tree-sitter for AST-aware chunking. Tika is overkill unless you're dealing with legacy enterprise formats (`.msg`, `.rtf`, old Office).

---

### 3. Chunking Strategy

All three agree semantic chunking beats fixed-size windows. But the depth of thinking varies.

**Deep Research** mentions both fixed-size-with-overlap and semantic splitting, references Databricks best practices, but doesn't differentiate by content type. It's "pick one strategy, apply everywhere."

**RAPTOR PDF** explicitly calls out "thematic breaks" — looking for semantic boundaries so chunks don't lose meaning. Good instinct, but it still treats all files the same way.

**My brainstorm** pushed hardest here: different strategies for prose (section/paragraph boundaries), code (AST-based at function/class level via tree-sitter), and config files (by top-level keys or embedded whole). This is the approach I'd still recommend — **uniform chunking is the single biggest silent accuracy killer** in these pipelines.

---

### 4. Embedding

| Dimension   | Deep Research                                      | RAPTOR PDF                                       | My Brainstorm                                |
| ----------- | -------------------------------------------------- | ------------------------------------------------ | -------------------------------------------- |
| Model       | OpenAI ada / Sentence Transformers (API-dependent) | BAAI/bge-m3 (local, multilingual, state-of-art)  | voyage-code-3 or nomic-embed-text (flexible) |
| Granularity | Chunk-level only                                   | Chunk-level + summary re-embedding (RAPTOR loop) | Chunk-level + file-level pooling             |

**The interesting divergence**: The RAPTOR approach embeds _summaries_ as new data points and clusters those. My brainstorm pools chunk embeddings to get file-level representations. These are solving the same problem (multi-granularity representation) from different directions.

**bge-m3 is a strong choice** for local-first. It handles code reasonably well and is genuinely state-of-the-art for retrieval tasks. If you're going fully local (which the RAPTOR plan commits to), bge-m3 is the right pick. If you're open to APIs, voyage-code-3 is better for code-heavy directories.

---

### 5. Clustering — The Central Disagreement

This is where the three approaches diverge most.

**Deep Research** is agnostic — it lists K-Means, DBSCAN, and GMM as options, suggests silhouette scores for validation, and moves on. No strong recommendation.

**RAPTOR PDF** commits fully to GMM and makes a real argument for it: soft clustering means a "Python Finance Scripts" file can be 80% Code and 20% Finance. It adds a confidence threshold (70–90%) and a retry loop — if BIC/silhouette is below 0.7, re-run with different K. This is a concrete, implementable design.

**My brainstorm** argued against GMM as the primary clustering method, flagging three problems: dimensionality (covariance estimation breaks with 768+ dim vectors), choosing K, and the Gaussian shape assumption on UMAP-reduced embeddings. I proposed HDBSCAN first (discovers clusters without K, handles noise) then GMM for soft assignment refinement.

**Reconciling these now — here's what I think the right answer is:**

The RAPTOR plan's retry loop is a genuinely good idea that my brainstorm was missing. The concept of "cluster, check score, adjust K, retry" is more robust than one-shot clustering. But the underlying concern about GMM on raw high-dimensional embeddings is still valid. Here's the synthesis:

1. **Reduce dimensionality first** (UMAP to ~20–30 dims) — non-negotiable regardless of clustering method
2. **Use HDBSCAN for initial cluster discovery** — it finds K for you and identifies noise/outlier chunks
3. **Use GMM for soft assignment** — initialize it with HDBSCAN's output, fit on the UMAP-reduced space
4. **Adopt the RAPTOR retry loop** — if silhouette < threshold, adjust `min_cluster_size` for HDBSCAN and re-run
5. **The RAPTOR recursion is the real differentiator** — summarize clusters, re-embed summaries, re-cluster. This builds the hierarchy naturally rather than imposing it

The recursion is the part worth stealing wholesale from the RAPTOR plan. Neither the deep research report nor my brainstorm had this, and it's the most architecturally elegant solution to "how do you go from flat chunks to a hierarchical project structure."

---

### 6. Structural Signals

**Deep Research**: Mentions metadata (paths, timestamps) in passing but never integrates it into clustering.

**RAPTOR PDF**: Completely ignores structural signals. The tree is built purely from semantic similarity. A file's directory location, its imports, its naming — none of that factors in.

**My brainstorm**: This was the strongest differentiator — import/reference graph analysis, directory co-location affinity, naming conventions, temporal proximity. Proposed as a post-clustering refinement step (Option B: cluster on embeddings first, then merge/split using structural signals).

**This matters a lot for accuracy.** Consider: you have `/project_a/utils/auth.py` and `/project_b/lib/auth_helper.py`. Semantically, their embeddings will be nearly identical — both discuss authentication. A purely semantic pipeline (RAPTOR or deep research) will cluster them together. But they belong to different projects. The import graph would show `project_a/utils/auth.py` is imported by `project_a/main.py` and has zero cross-references to project_b. That's the signal that separates them.

**My recommendation**: The RAPTOR recursive tree is the right core architecture, but it needs a structural signal injection layer between the embedding step and the clustering step. Without it, you'll get topically coherent but project-incoherent clusters.

---

### 7. Summarization

**Deep Research** describes map-reduce (chunk summaries → cluster summary) and cites a code-repo study using 5k-token chunks → summarize → re-summarize. Also mentions blended extractive/abstractive.

**RAPTOR PDF** bakes summarization into the recursive loop itself — each cluster gets a 100-word executive summary that becomes a parent node. Summarization isn't a final step; it's part of the tree-building process.

**My brainstorm** suggested hierarchical map-reduce as a final step, separate from clustering.

**The RAPTOR approach is better here.** Making summarization part of the recursion means each level of the tree has a human-readable summary. You don't just get a final output — you get summaries at every granularity level. The deep research report's map-reduce is essentially what RAPTOR does, but RAPTOR makes the intermediate summaries _reusable_ as data points.

---

### 8. Validation

**Deep Research** has the most thorough validation section: ROUGE scores, factual consistency checks, fluency/coherence/relevance evaluation, prompt strategy testing (zero-shot vs few-shot).

**RAPTOR PDF** has the confidence loop (BIC/silhouette threshold → retry), which validates clustering quality, but doesn't validate summary quality.

**My brainstorm** suggested LLM-based borderline review (for chunks with low cluster confidence) but didn't go deep on summary validation.

**You need both**: clustering validation (RAPTOR's retry loop) AND summary validation (deep research's factual consistency checks). These are orthogonal. Good clusters can still produce hallucinated summaries, and bad clusters can't be saved by good summarization.

---

## The Synthesized Architecture

Taking the best from each:

```
RAPTOR's recursive tree (core architecture)
  + My brainstorm's structural signal injection (accuracy)
  + My brainstorm's type-aware chunking (code vs docs)
  + RAPTOR's local-first stack (Ollama + bge-m3 + ChromaDB)
  + HDBSCAN → GMM hybrid (robust cluster discovery + soft assignments)
  + RAPTOR's confidence retry loop (self-correcting)
  + Deep research's summary validation (factual consistency)
```

The RAPTOR recursion is the right backbone. The gap it has is that it's **purely semantic** — it ignores structural signals that are critical for distinguishing same-topic-different-project files. Your biggest accuracy risk isn't the clustering algorithm; it's the **input signal quality**. Embeddings alone aren't enough.

What's your stance on local-first vs API-based? The RAPTOR plan commits to full local (Ollama + Llama 3.1), which is great for privacy and cost but means your summarization quality ceiling is lower than GPT-4/Claude. That tradeoff is worth discussing before you start building.

<hr>

Yes, the scattered files problem is exactly what this pipeline is built to solve — and it's worth going deeper on _why_ it's hard and _how_ each layer contributes to getting it right.

## The Scattered Files Problem, Concretely

Imagine this real directory:

```
/shared_drive/
  ├── john_docs/
  │   ├── Q3_budget_draft.xlsx
  │   └── random_notes.txt
  ├── presentations/
  │   ├── client_pitch_v2.pptx
  │   └── onboarding_deck.pptx
  ├── dev/
  │   ├── api/
  │   │   └── auth_service.py
  │   └── scripts/
  │       └── data_migration.py
  ├── old_stuff/
  │   ├── Q3_budget_FINAL.xlsx
  │   └── client_pitch_old.pptx
  └── meeting_notes/
      └── 2024-03-15_kickoff.md
```

A human instantly sees: the Q3 budget files are the same project despite being in `john_docs/` and `old_stuff/`. The client pitch deck and the kickoff meeting notes are related. `auth_service.py` might be referenced in the onboarding deck. But a naive semantic-only pipeline might cluster all `.xlsx` files together (both budget files + unrelated spreadsheets) because spreadsheet text embeds similarly, or it might miss that the meeting notes mention the API auth work.

Three things need to go right for the system to correctly reunite scattered files:

**1. Parsing must preserve identity, not just content.** When you parse `Q3_budget_draft.xlsx` and `Q3_budget_FINAL.xlsx`, the raw text might be nearly identical. The filenames, timestamps, and directory paths are what tell you these are versions of the same thing, not two separate projects. This is why the file registry layer matters — you need to capture and carry forward metadata as first-class features, not discard it after parsing.

**2. Embeddings must work cross-format.** A `.pptx` slide saying "Auth Service Architecture" and a `.py` file implementing that architecture need to land near each other in embedding space despite being completely different formats. This is where bge-m3 earns its keep — it's trained on diverse text types. But it's also why chunking matters: if you chunk the pptx slide-by-slide and the Python file function-by-function, you're comparing semantic units of similar granularity. If you chunk the pptx as one giant blob and the Python file in 200-token windows, the embeddings won't align well.

**3. Clustering must tolerate format-mixing within clusters.** This is the argument for soft clustering. A meeting notes file might be 60% about the client pitch project and 40% about the API migration project because both were discussed in the same meeting. Hard clustering forces a choice; GMM's soft assignment lets it contribute to both clusters proportionally.

## The Local-First Stack — Refined

Since privacy is non-negotiable, let's lock down the stack with specific model choices and justify each:

**Embedding: bge-m3** — agreed with the RAPTOR plan. It's the best local embedding model right now for mixed-content scenarios. Handles multilingual too if that matters. Runs comfortably on CPU, no GPU required for inference (though GPU speeds it up). One thing to note: bge-m3 supports multiple retrieval modes (dense, sparse, multi-vector). For clustering you want the dense vectors. But the sparse vectors could be useful later for keyword-based retrieval within clusters.

**LLM: Llama 3.1 8B via Ollama for development, 70B for production.** The 8B is fast enough for iterating on prompts and testing the pipeline. But for final summary quality, the 70B is meaningfully better at synthesizing diverse source material into coherent prose. If you have a machine with 48GB+ VRAM (or can do CPU inference with patience), use 70B. Alternatively, **Qwen 2.5 72B** is worth benchmarking against Llama 3.1 70B — it's been strong on structured output and instruction following, which matters for your summary generation prompts.

**Vector store: ChromaDB** — fine for this use case. You're doing batch processing, not real-time retrieval at scale. ChromaDB's persistence is simple and local. You won't need Pinecone or Weaviate's distributed features. One consideration: ChromaDB stores embeddings + metadata together, which means you can filter by metadata at query time (e.g., "give me all chunks from `.py` files in this cluster"). That's useful for the structural refinement step.

**Dimensionality reduction: UMAP** — runs locally, no API needed. `umap-learn` in Python.

**Clustering: scikit-learn** for GMM, `hdbscan` library for HDBSCAN. Both fully local.

## How the Scattered File Problem Gets Solved Layer by Layer

Let me walk through how a scattered file (`client_pitch_v2.pptx` in `/presentations/` and `client_pitch_old.pptx` in `/old_stuff/`) gets correctly reunited:

**Layer 1 — File Registry:** Both files are logged with their full paths, extensions, sizes, modification dates. The system notes they share a naming pattern (`client_pitch_*`). This similarity score gets stored as metadata.

**Layer 2 — Parsing:** Unstructured.io extracts slide text from both pptx files. Each slide becomes a chunk with provenance: `{source: "presentations/client_pitch_v2.pptx", slide: 3, type: "pptx"}`.

**Layer 3 — Embedding:** Chunks from both files embed nearby because they discuss the same client, same proposal, same deliverables. But so might chunks from `meeting_notes/2024-03-15_kickoff.md` if that meeting discussed the client pitch.

**Layer 4 — UMAP + HDBSCAN:** Initial clustering groups these chunks together. The meeting notes chunks that discussed the pitch also land in this cluster. The meeting notes chunks that discussed the API migration land in a different cluster. A chunk that discussed both gets low confidence from HDBSCAN and gets flagged.

**Layer 5 — GMM refinement:** The flagged chunk gets soft-assigned: 55% client pitch cluster, 40% API migration cluster. Both clusters now "know about" this cross-cutting content.

**Layer 6 — Structural refinement:** The system checks — do `client_pitch_v2.pptx` and `client_pitch_old.pptx` share naming patterns? Yes. Are they in different directories? Yes. This confirms they're versions of the same project artifact, not separate projects. The system also checks if the kickoff meeting notes reference the client pitch by name — if so, the cross-reference strengthens the cluster assignment.

**Layer 7 — RAPTOR recursion:** The client pitch cluster gets summarized. That summary gets re-embedded and in the next recursion level, it might cluster with a "sales pipeline" cluster, forming a higher-level "Business Development" category.

**Layer 8 — Output:** The final markdown file has a section for the client pitch project that references all three source files across their scattered locations, with the summary drawing from all of them.

## The One Gap We Need to Design For

There's a scenario none of the three approaches handles well: **files that belong to the same project but have zero semantic overlap**. Example: a project has a `requirements.txt`, a `Dockerfile`, a `schema.sql`, and a `README.md`. The README might semantically relate to the Python code, but the Dockerfile and schema.sql are semantically distant from everything — they'll embed in completely different regions.

The **import/dependency graph** is the only thing that saves you here. `requirements.txt` lists the packages used by `.py` files in the same directory. `Dockerfile` references the app entrypoint. `schema.sql` is referenced in a database config file. These are structural links, not semantic ones.

For the first version, I'd suggest a pragmatic rule: **files that share a common parent directory within 2 levels and have no strong cluster assignment elsewhere get grouped with the dominant cluster of their directory neighborhood.** This is a heuristic, not an algorithm, but it catches the common case of config/infra files that semantically orphan but structurally belong.

## Next Steps

Before building, there are three things worth deciding:

First, **what's the minimum viable test case?** Pick a real messy directory you have access to, ideally one where you already know what the "right" groupings should be. This becomes your ground truth for evaluating accuracy. Without it you'll be tuning blind.

Second, **recursion depth.** The RAPTOR plan recurses until a single root summary. For a small directory (50–100 files), you might only need 2–3 levels. For thousands of files, you might need 5+. Do you want a fixed depth or a dynamic stop condition (e.g., stop when the top-level has fewer than N clusters)?

Third, **output format.** The RAPTOR plan outputs a regenerated directory structure + README per folder. Your original description was a single compiled markdown file per project. These are different deliverables. The single-md-per-project approach is simpler and probably more useful for your initial use case — you can always add directory regeneration later. Which do you want to build first?

<hr>

Good, these clarifications tighten the design significantly. Let me address each one and refine the architecture.

## 1. The "Directory Neighborhood" Rule for Supporting Files

Your instinct is right — a `Dockerfile`, `schema.sql`, `requirements.txt`, `.env.example` sitting alongside `.py` files shouldn't be independently clustered. They're satellite files. The logic needs to be:

**Before clustering even begins**, run a "project root detection" pass. The idea is that certain files are **anchor files** — they signal "this directory is a project root." Things like `package.json`, `requirements.txt`, `pyproject.toml`, `Cargo.toml`, `pom.xml`, `Makefile`, `docker-compose.yml`, `go.mod`. When you find one of these, you tag every file in that directory subtree as belonging to a **pre-cluster group**.

Then the pipeline has two modes for how it treats files:

**Mode A — Anchored files.** Files inside a detected project root get a "project affinity tag" before they ever hit the embedding layer. They still get embedded and chunked individually (so the summary captures what each file does), but during clustering they carry a strong prior that they belong together. The clustering can still override this if the semantic signal is overwhelming (e.g., a completely unrelated file dumped in the wrong folder), but the default is "same project root = same cluster."

**Mode B — Unanchored files.** Files outside any detected project root (loose documents, random spreadsheets, scattered notes) get no prior. They're purely dependent on semantic clustering to find their group. This is where the pipeline earns its value — grouping a budget spreadsheet in `/john_docs/` with a budget presentation in `/old_stuff/`.

This two-mode approach means you're not trying to force a single algorithm to handle both "this is clearly one project's codebase" and "these are random files that might be topically related." Different problems, different treatment.

For non-code project directories, the anchor files would be different — maybe a directory with multiple `.docx` and `.pptx` files sharing a naming convention, or a folder explicitly named after a project. The detection heuristic for work files is softer, but things like shared prefixes in filenames (`Q3_budget_draft.xlsx`, `Q3_budget_final.xlsx`, `Q3_budget_notes.docx`) or a README/index file can serve the same anchoring purpose.

## 2. The Ignore List — Non-Negotiable for Sanity

This needs to be baked in at the file discovery layer, before anything gets parsed. A single `node_modules` folder can contain 50,000+ files that would obliterate your pipeline.

The ignore list should work like `.gitignore` — pattern-based, with sane defaults that can be extended. The defaults:

**Package/dependency directories:** `node_modules/`, `vendor/`, `bower_components/`, `.gradle/`, `target/` (Java/Rust), `__pycache__/`, `*.pyc`, `.eggs/`, `*.egg-info/`, `site-packages/`

**Virtual environments:** `venv/`, `.venv/`, `env/`, `.env/` (but NOT `.env` files — those are config), `conda-envs/`

**Build artifacts:** `dist/`, `build/`, `out/`, `.next/`, `.nuxt/`, `*.min.js`, `*.min.css`, `*.map`, `*.bundle.js`

**Cache/temp:** `.cache/`, `.tmp/`, `*.tmp`, `*.swp`, `*.bak`, `.DS_Store`, `Thumbs.db`, `*.log` (debatable — some logs are informative)

**Version control:** `.git/`, `.svn/`, `.hg/`

**IDE/editor:** `.idea/`, `.vscode/` (debatable — `settings.json` can be project-relevant), `*.suo`, `*.user`

**Binary/media that can't be meaningfully embedded:** `*.jpg`, `*.png`, `*.gif`, `*.mp4`, `*.mp3`, `*.zip`, `*.tar.gz`, `*.exe`, `*.dll`, `*.so` — unless you add an image description layer later

Additionally, the user should be able to drop a `.regenai-ignore` file (or whatever you name the tool) in the root directory to add custom patterns. Some organizations will have specific folders to skip.

**The implementation detail that matters:** this filtering happens in the file registry step, before any parsing. You walk the tree, check every path against the ignore patterns, and only register files that pass. This means your file count of 200–500 is the _post-filter_ count, which is very manageable.

## 3. Output Structure

Clear. Here's what the output directory looks like:

```
/output/
  ├── Project_Alpha/
  │   └── Project_Alpha.md
  ├── Q3_Financial_Planning/
  │   └── Q3_Financial_Planning.md
  ├── API_Migration/
  │   └── API_Migration.md
  └── _uncategorized/
      └── _uncategorized.md
```

Each `.md` file contains:

**Header section** — project/topic name (generated by LLM from cluster content), confidence score, number of source files.

**Source files section** — a list of every file that was clustered into this project, with original paths. This is critical for traceability — you need to know _where_ the system pulled from.

**Summary section** — the hierarchical summary. For code projects this covers architecture, key components, dependencies. For work files this covers key themes, decisions, deliverables, timelines mentioned.

**Cross-references section** — if chunks from this cluster had soft membership in other clusters (the GMM confidence split), note which other projects this one overlaps with and why. This captures the "meeting notes that discussed two projects" case.

The `_uncategorized` folder catches noise — files that HDBSCAN flagged as outliers, chunks that didn't cluster with confidence above threshold. Better to surface these honestly than force them into a bad cluster.

## 4. Handling "Normal Work Files" — Not Just Code

This is an important framing shift. The pipeline needs to be equally good at clustering a directory that looks like:

```
/shared_drive/
  ├── HR_stuff/
  │   ├── new_hire_checklist.docx
  │   └── benefits_summary.pdf
  ├── random/
  │   ├── team_offsite_ideas.md
  │   ├── vendor_comparison.xlsx
  │   └── NDA_template.docx
  ├── project_phoenix/
  │   ├── proposal_v1.pptx
  │   └── timeline.xlsx
  └── from_email/
      ├── phoenix_feedback_from_client.pdf
      └── budget_approval_scan.pdf
```

No anchor files here. No `requirements.txt` to detect project roots. The entire grouping has to come from semantic similarity + naming patterns + temporal proximity.

This means the pipeline needs to be robust in **Mode B** (unanchored), not just Mode A (code projects). For work files, the clustering signal hierarchy is:

1. **Content similarity** (strongest) — do these files discuss the same topics, people, deliverables?
2. **Naming patterns** — `phoenix_*`, `Q3_*`, `NDA_*`
3. **File type grouping** (weakest, use as tiebreaker only) — don't cluster all PDFs together just because they're PDFs, but if two files have similar content AND similar format, that's a mild reinforcement

The LLM labeling step also needs to adapt. For code projects, cluster labels might be "Authentication Service" or "Data Pipeline." For work files, labels might be "Project Phoenix — Client Proposal" or "HR Onboarding Materials." The summarization prompts need to handle both modes — one prompt template for code-heavy clusters, another for document-heavy clusters, detected by the file type distribution within the cluster.

## Revised Architecture — Final Pre-Implementation View

```
INPUT: directory path
         │
    ┌────▼─────┐
    │  File     │  Walk tree, apply ignore patterns
    │  Registry │  Classify file types, extract metadata
    └────┬─────┘  Detect project roots (anchor files)
         │
    ┌────▼─────┐
    │  Parse    │  Unstructured.io for docs/pptx/pdf/xlsx
    │  Layer    │  tree-sitter for code files
    └────┬─────┘  Raw text for .md/.txt/.json/.yaml
         │
    ┌────▼──────┐
    │  Chunk    │  Semantic chunking for prose
    │  Layer    │  AST-based for code
    └────┬──────┘  Attach provenance metadata to every chunk
         │
    ┌────▼──────────┐
    │  Embed         │  bge-m3 (local)
    │  (+ metadata)  │  Chunk-level embeddings
    └────┬──────────┘  Store in ChromaDB with metadata
         │
    ┌────▼──────────┐
    │  Pre-cluster   │  Group anchored files (project roots)
    │  Grouping      │  Name-pattern detection
    └────┬──────────┘  Inject affinity scores
         │
    ┌────▼──────┐
    │  UMAP     │  Reduce to ~20-30 dims
    │  Reduce   │  Preserves topological structure
    └────┬──────┘
         │
    ┌────▼──────────────────────────┐
    │  RAPTOR Recursive Loop        │
    │  ┌─────────────────────────┐  │
    │  │ HDBSCAN → discover K    │  │
    │  │ GMM → soft assignments  │  │
    │  │ Confidence check        │──│──► If below threshold,
    │  │ (silhouette/BIC)        │  │    adjust & retry
    │  └──────────┬──────────────┘  │
    │  ┌──────────▼──────────────┐  │
    │  │ Summarize each cluster  │  │
    │  │ (Ollama / Llama 3.1)    │  │
    │  └──────────┬──────────────┘  │
    │  ┌──────────▼──────────────┐  │
    │  │ Re-embed summaries      │  │
    │  │ Re-cluster (next level) │  │
    │  └──────────┬──────────────┘  │
    │             │ Stop when       │
    │             │ N clusters < T  │
    └─────────────┴─────────────────┘
         │
    ┌────▼──────────┐
    │  Structural    │  Merge/split using import graphs,
    │  Refinement    │  directory affinity, naming patterns
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │  Validation    │  Factual consistency check
    │  Layer         │  Coverage verification
    └────┬──────────┘
         │
    ┌────▼──────────┐
    │  Output        │  One directory per project/topic
    │  Generator     │  One .md file per project
    └───────────────┘  _uncategorized for outliers

OUTPUT: /output/ directory with project summaries
```

## Before You Build the Test Directory

A few things that'll make the test useful. Include at least three distinct "projects" with 2–3 files each scattered across different subdirectories. Include at least one file that genuinely belongs to two projects (like meeting notes covering multiple topics). Include a few completely orphaned files that don't belong to anything. Include at least one code project with supporting files (Dockerfile, config, etc.). And include a couple of version duplicates (draft/final of the same document).

That gives you enough variety to stress-test every layer of the pipeline. Once you've got the directory set up, we can start implementing module by module — file registry first, then parsing, then embedding, and build up to clustering iteratively.

<hr>

Good. Let's lay out the full implementation plan, module by module, in the order we'll build them. Each module is independently testable before moving to the next.

## Project Structure

```
regenai/
  ├── pyproject.toml
  ├── README.md              (later)
  ├── .python-version         (3.11)
  ├── src/
  │   └── regenai/
  │       ├── __init__.py
  │       ├── main.py              (CLI entrypoint & orchestrator)
  │       ├── registry.py          (Module 1: file discovery & metadata)
  │       ├── ignore.py            (Module 1b: ignore pattern engine)
  │       ├── parser.py            (Module 2: file parsing)
  │       ├── chunker.py           (Module 3: type-aware chunking)
  │       ├── embedder.py          (Module 4: embedding layer)
  │       ├── clusterer.py         (Module 5: UMAP + HDBSCAN + GMM)
  │       ├── raptor.py            (Module 6: recursive summarize-recluster loop)
  │       ├── refiner.py           (Module 7: structural refinement)
  │       ├── summarizer.py        (Module 8: final summary generation)
  │       ├── validator.py         (Module 9: factual consistency checks)
  │       ├── output.py            (Module 10: markdown generation)
  │       └── config.py            (shared constants, thresholds, model names)
  ├── tests/
  │   └── ...
  └── .regenai-ignore              (default ignore patterns)
```

The `src/regenai/` layout is standard for `uv` + `pyproject.toml` projects. Each module has a single responsibility and a clean interface to the next.

## Dependencies

Here's what we need and why:

**Core parsing:**

- `unstructured[local-inference]` — parses pptx, pdf, docx, xlsx into text. The local-inference extra avoids any API calls. Heaviest dependency but nothing else handles this breadth of formats locally.
- `tree-sitter` + language grammars (`tree-sitter-python`, `tree-sitter-javascript`, etc.) — AST-based chunking for code. We only install grammars for languages we detect in the directory.

**Embeddings:**

- `sentence-transformers` — loads and runs bge-m3 locally. Pulls in torch as a transitive dependency, which is large but unavoidable for local embeddings.
- `chromadb` — persistent vector store. Stores embeddings + metadata together, supports metadata filtering.

**Clustering:**

- `umap-learn` — dimensionality reduction before clustering.
- `hdbscan` — density-based initial clustering.
- `scikit-learn` — GMM for soft assignments, silhouette scoring, BIC.

**LLM:**

- `ollama` (Python client) — talks to the local Ollama server. Assumes Ollama is installed and running separately with Llama 3.1 pulled.

**Utilities:**

- `pathspec` — `.gitignore`-style pattern matching for the ignore engine. Battle-tested, same library `gitignore` uses.
- `rich` — terminal output, progress bars, logging. Not strictly necessary but makes debugging the pipeline much easier when you're processing hundreds of files.

**One dependency question for you:** `tree-sitter` requires building C extensions for each language grammar. An alternative is `pygments` for lighter-weight code tokenization — it won't give you full AST structure but it can identify function/class boundaries in most languages with much less setup friction. For a first pass, we could use `pygments` for code boundary detection and upgrade to `tree-sitter` later if the accuracy isn't sufficient. Thoughts?

## Module Implementation Order & Interfaces

### Module 1: `registry.py` + `ignore.py`

**What it does:** Walks the input directory, filters out ignored paths, classifies every file, and builds a structured registry.

**Input:** A directory path string.

**Output:** A list of `FileEntry` objects, each containing:

```python
@dataclass
class FileEntry:
    path: Path              # absolute path
    relative_path: Path     # relative to input root
    extension: str
    file_type: str          # "code", "document", "config", "data", "unknown"
    language: str | None    # for code: "python", "javascript", etc.
    size_bytes: int
    modified_at: datetime
    parent_dir: str         # immediate parent directory name
    depth: int              # directory depth from root
    project_root: str | None  # detected project root, if anchored
    name_prefix: str | None   # extracted naming pattern (e.g., "Q3_budget")
```

**Key logic:**

- `ignore.py` loads patterns from `.regenai-ignore` (shipped defaults + user overrides), uses `pathspec` to match. Checked against every discovered path _before_ any file I/O.
- Project root detection: scan for anchor files (`package.json`, `requirements.txt`, `pyproject.toml`, `Cargo.toml`, `Makefile`, `docker-compose.yml`, `go.mod`, `pom.xml`). Every file under a detected root gets tagged with that root path.
- Name prefix extraction: strip extension, strip version suffixes (`_v2`, `_final`, `_draft`, `_old`), normalize. Two files with the same prefix in different directories get flagged as potential versions/duplicates.
- File type classification is extension-based with a lookup table — nothing fancy needed here.

**Testable independently:** Run it on your test directory, print the registry, verify the classifications and project root detections look right before moving on.

---

### Module 2: `parser.py`

**What it does:** Takes a `FileEntry`, extracts text content.

**Input:** A `FileEntry` object.

**Output:** A `ParsedFile` object:

```python
@dataclass
class ParsedFile:
    entry: FileEntry
    raw_text: str
    sections: list[str] | None    # for docs with clear sections
    metadata: dict                 # anything the parser extracts (author, title, slide count, etc.)
    parse_success: bool
    error: str | None
```

**Key logic:**

- Routes to the right parser based on `file_type`:
  - Documents (pdf, docx, pptx, xlsx): `unstructured.io` partition functions
  - Code files: direct read with encoding detection (`chardet` if needed, usually utf-8)
  - Markdown/text: direct read
  - Config (yaml, json, toml): direct read (these are small, pass through as-is)
- For pptx specifically: preserve slide boundaries as section markers — this matters for chunking
- For xlsx: extract sheet names and cell content as structured text, not just a raw dump
- Graceful failure: if a file can't be parsed, log the error, set `parse_success = False`, and continue. One corrupt PDF shouldn't kill the pipeline.

---

### Module 3: `chunker.py`

**What it does:** Splits parsed text into semantically meaningful chunks with provenance metadata.

**Input:** A `ParsedFile` object.

**Output:** A list of `Chunk` objects:

```python
@dataclass
class Chunk:
    id: str                    # unique chunk ID (file_hash + chunk_index)
    text: str
    source_file: Path
    file_type: str
    chunk_index: int
    total_chunks: int          # how many chunks this file produced
    project_root: str | None   # inherited from FileEntry
    name_prefix: str | None    # inherited from FileEntry
    metadata: dict             # any additional context
```

**Key logic:**

- **Document/prose chunking:** Split at paragraph or section boundaries. If sections exist (from parser), use those as primary boundaries. If a section exceeds ~800 tokens, sub-split at paragraph breaks. Overlap of ~100 tokens between adjacent chunks.
- **Code chunking:** This is where the `pygments` vs `tree-sitter` decision matters. Either way, the goal is: split at function/class/method boundaries. If a function exceeds ~800 tokens, keep it whole (long functions are still one semantic unit — splitting mid-function produces garbage). If a file is under ~800 tokens total, don't chunk it at all.
- **Config/small files:** If under ~500 tokens, embed as a single chunk. No splitting needed.
- **Provenance is critical:** Every chunk must carry enough metadata to trace it back to its source file and position. This feeds into the output markdown later.

---

### Module 4: `embedder.py`

**What it does:** Converts chunks to vector embeddings, stores in ChromaDB.

**Input:** A list of `Chunk` objects.

**Output:** ChromaDB collection populated with embeddings + metadata.

**Key logic:**

- Load bge-m3 via `sentence-transformers`. First run downloads the model (~2.3GB), subsequent runs load from cache.
- Batch embedding — don't embed one chunk at a time. Process in batches of 32–64 depending on available RAM.
- Store in ChromaDB with the full metadata dict from the `Chunk` object. This means you can later filter by `file_type`, `project_root`, `name_prefix`, etc.
- Also compute file-level embeddings: mean-pool all chunk embeddings for each file and store separately. These are used for coarser-grained clustering if needed.

---

### Module 5: `clusterer.py`

**What it does:** The core algorithmic layer. UMAP reduction → HDBSCAN → GMM refinement.

**Input:** Embedding vectors + metadata from ChromaDB.

**Output:** Cluster assignments with confidence scores:

```python
@dataclass
class ClusterAssignment:
    chunk_id: str
    primary_cluster: int
    confidence: float              # GMM probability for primary
    secondary_cluster: int | None  # if soft assignment is significant
    secondary_confidence: float | None
    is_noise: bool                 # HDBSCAN outlier flag
```

**Key logic:**

- Pull all embeddings from ChromaDB as a numpy array.
- UMAP reduction to ~25 dimensions. Key params: `n_neighbors=15`, `min_dist=0.1`, `metric='cosine'`. These are starting values — we'll tune based on your test directory.
- HDBSCAN with `min_cluster_size` as a function of total chunk count. For 200–500 files producing maybe 1000–3000 chunks, `min_cluster_size=10` is a reasonable start. `prediction_data=True` to enable soft clustering later.
- Extract cluster labels and noise flags from HDBSCAN.
- Fit GMM on the UMAP-reduced space, initialized with HDBSCAN's K. Use `covariance_type='tied'` or `'diag'` to avoid the covariance estimation problem (full covariance with ~25 dims and ~2000 points is borderline — `diag` is safer).
- Compute silhouette score. If below 0.5, adjust `min_cluster_size` up or down and re-run. Cap retries at 5.
- For each chunk, extract primary and secondary cluster assignments from GMM posteriors. Secondary assignment only recorded if its probability exceeds 0.15 (otherwise it's noise).

---

### Module 6: `raptor.py`

**What it does:** The recursive loop — summarize clusters, re-embed summaries, re-cluster at the next level.

**Input:** Cluster assignments + chunks.

**Output:** A tree structure of clusters with summaries at each level.

**Key logic:**

- For each cluster at level 0, gather all chunk texts and feed to Ollama with a summarization prompt. Prompt varies by cluster composition:
  - Code-heavy cluster: "Summarize the architecture, key functions, and purpose of this codebase component."
  - Document-heavy cluster: "Summarize the key topics, decisions, and deliverables covered in these documents."
  - Mixed: "Summarize what this collection of code and documentation covers, focusing on the project's purpose and key components."
- Each summary becomes a new "chunk" at level 1. Re-embed these summaries.
- Re-cluster the summary embeddings (UMAP + HDBSCAN + GMM again, but with smaller data so params adjust accordingly).
- Repeat until the number of clusters at a level is ≤ 3, or until two consecutive levels produce the same groupings.
- Store the full tree: `{level: 0, clusters: [...], level: 1, clusters: [...], ...}`.

---

### Modules 7–10 (Refiner, Summarizer, Validator, Output)

These are downstream of the core pipeline. Let's design them in detail after modules 1–6 are working, since their exact interface depends on what the clustering actually produces on your test data. But in brief:

- **Refiner:** Post-processing that checks structural signals (import graphs, naming patterns, directory proximity) and proposes merge/split operations on clusters. Presents these as suggestions with reasoning — you approve or reject before they're applied.
- **Summarizer:** Takes the RAPTOR tree and generates the final detailed summary per project. This is the hierarchical map-reduce step — chunk summaries → cluster summary → project summary.
- **Validator:** Compares claims in the summary against source chunks. Flags any statement that can't be traced back to a chunk. Uses embedding similarity between summary sentences and source chunks as a proxy for factual grounding.
- **Output:** Generates the markdown files and directory structure. Templates for the md format, handles cross-reference sections.

## Implementation Sequence

I'd suggest building in this order, testing each module before moving to the next:

1. `config.py` + `ignore.py` + `registry.py` — get file discovery working
2. `parser.py` — verify text extraction on your test files
3. `chunker.py` — verify chunk quality, especially for code vs docs
4. `embedder.py` — get embeddings into ChromaDB, do a quick similarity sanity check
5. `clusterer.py` — this is the hardest module, iterate on params with your test data
6. `raptor.py` — the recursive loop, depends on Ollama being set up
7. Modules 7–10 — downstream processing

Questions before we start coding:

**On tree-sitter vs pygments** — pygments is lighter but less precise. For a first pass with your test directory, which has a known number of code files, pygments is probably fine. Agree?

**On Ollama setup** — do you already have Ollama installed and a model pulled, or do we need to account for setup time? Which Llama 3.1 variant are you planning to run (8B vs 70B)? What's your GPU situation?

**On the orchestrator** — do you want a CLI (`python -m regenai /path/to/messy/dir`) or are you thinking of a different entrypoint?

<hr>

Alright, let me address each point honestly, especially the hardware situation because that changes some decisions.

## Hardware Reality Check

Your Quadro P2000 has 5GB dedicated VRAM. The "20.8 GB" Windows reports includes shared system RAM which CUDA can't use effectively for ML inference. So we're working with:

- **5GB VRAM** (dedicated, actual usable GPU memory)
- **32GB system RAM** (plenty for CPU inference)
- **Xeon W-2133** — 6 cores, solid for CPU workloads

What this means for model choices:

**Llama 3.1 70B is off the table.** Even Q4 quantized, it needs ~40GB. Not happening on this hardware, not even with CPU offloading — 32GB RAM would be consumed entirely by the model weights leaving nothing for the OS, the embedding model, ChromaDB, or the rest of the pipeline.

**Llama 3.1 8B Q4 is ~4.7GB** — it technically fits in 5GB VRAM but it's razor-thin. One large context window and you're spilling to CPU. It'll work but expect slower generation when context gets heavy (which it will during cluster summarization when you're feeding 10+ chunks).

**The practical approach:** Run the 8B model and let Ollama handle the GPU/CPU split automatically. Ollama is smart about partial offloading — it'll put as many layers on GPU as fit and run the rest on CPU. With 32GB RAM, CPU inference on 8B is perfectly usable, just slower (~10–15 tokens/sec instead of ~40). Since this is a batch pipeline and not an interactive chat, speed is secondary to accuracy.

**The "mix of models" idea I mentioned earlier** was about using 8B for development iteration and 70B for production runs. Since 70B isn't feasible on your hardware, the alternative is: use 8B locally for all pipeline runs, and if you ever need higher summary quality for a specific batch, you could do a one-time run against a hosted 70B via API (OpenRouter with Llama 3.1 70B, for example). But that breaks the local-only privacy constraint, so it's only relevant for non-sensitive directories. For now, **8B is your model, and we'll optimize prompts to get the best summaries out of it.**

One more option worth mentioning: **Qwen 2.5 7B or Mistral 7B** — both are competitive with Llama 3.1 8B on summarization tasks and sometimes better on structured output. We can benchmark all three on your test directory once the pipeline is running and pick the winner. Swapping models in Ollama is one command.

## Ollama Setup — Checking & Pulling

Run these in your terminal:

```bash
# Check if Ollama is running
ollama --version

# See what models you already have
ollama list

# Pull Llama 3.1 8B (this is the one we want)
ollama pull llama3.1:8b

# Optionally pull alternatives to benchmark later
ollama pull qwen2.5:7b
ollama pull mistral:7b

# Quick test — make sure it responds
ollama run llama3.1:8b "Summarize what a Dockerfile does in 2 sentences."
```

The `pull` command downloads the model — Llama 3.1 8B is about a 4.7GB download. Run `ollama list` after to confirm it's there. The `run` command is just a sanity check.

**Important:** When the pipeline is running, Ollama needs to be running as a server in the background. On most systems `ollama serve` starts it, or it runs as a system service after installation. The Python `ollama` library connects to `http://localhost:11434` by default.

## Tree-sitter — Since You Want Accuracy

Tree-sitter gives you proper AST parsing, which means we can chunk at exact function/class boundaries rather than guessing from indentation or regex. The tradeoff is setup complexity — each language needs a compiled grammar. Here's how we'll handle it:

**Auto-detect languages from the file registry**, then only install the grammars we need. If your test directory has Python and JavaScript, we install `tree-sitter-python` and `tree-sitter-javascript`. No need to pre-install every language.

The supported languages we'll target initially: Python, JavaScript/TypeScript, Java, C/C++, Go, Rust. These cover the vast majority of codebases. If a code file is in a language we don't have a grammar for, we fall back to a simpler heuristic splitter (regex-based function detection). Clean degradation, not failure.

`tree-sitter` bindings for Python have improved recently — `tree-sitter` 0.21+ uses a much simpler API than the old one. We'll use `tree-sitter` with `tree-sitter-languages` (a package that bundles pre-compiled grammars for ~15 languages) to avoid the build-from-source pain.

## CLI Design

Here's what I'm thinking for the entrypoint:

```bash
# Option 1: Pass directory as argument
regenai /path/to/messy/directory

# Option 2: No argument — interactive prompt
regenai
# → "Enter the directory path to analyze: "
# → user types or pastes path
# → optional: show file count and ask for confirmation before proceeding

# Option 3: With output directory override
regenai /path/to/messy/dir --output /path/to/output

# Other useful flags
regenai /path/to/dir --model llama3.1:8b    # override default model
regenai /path/to/dir --dry-run               # scan and show file registry, don't process
regenai /path/to/dir --verbose               # detailed logging
```

The `--dry-run` flag is particularly useful — it runs Module 1 (file discovery) only and shows you what the pipeline would process, so you can verify the ignore patterns are working and the file classifications look right before committing to a full run.

Implementation: `argparse` for argument parsing, `rich` for the interactive prompts and progress display. If no path argument is given, prompt with `rich.prompt.Prompt` and validate the path exists before proceeding.

The entrypoint in `pyproject.toml`:

```toml
[project.scripts]
regenai = "regenai.main:cli"
```

So after `uv pip install -e .`, the `regenai` command is available globally in the venv.

## Revised Dependency List

Based on all decisions:

```
# Core
unstructured[local-inference]    # document parsing
tree-sitter                      # AST parsing for code
tree-sitter-languages            # pre-compiled grammars

# Embeddings & Storage
sentence-transformers            # loads bge-m3 locally
chromadb                         # persistent vector store
torch                            # transitive dep of sentence-transformers

# Clustering
umap-learn                       # dimensionality reduction
hdbscan                          # density-based clustering
scikit-learn                     # GMM, silhouette, BIC

# LLM
ollama                           # Python client for local Ollama

# Utilities
pathspec                         # .gitignore-style pattern matching
rich                             # CLI output, progress bars, prompts
chardet                          # encoding detection for text files
```

**One concern:** `unstructured[local-inference]` pulls in a lot of transitive dependencies (including some ML libraries for PDF/image processing). The full install can be 2–3GB. An alternative is `unstructured` without the local-inference extra and manually installing only the format-specific deps we need (`python-pptx`, `python-docx`, `openpyxl`, `pdfplumber`). This is lighter but means we're writing more parser boilerplate ourselves. For a first pass, the lighter approach might be better — we probably only need pptx, docx, pdf, and xlsx parsing, and dedicated libraries for each are simpler and more predictable than the full Unstructured stack.

What's your preference: full `unstructured[local-inference]` for maximum format coverage, or the lighter approach with individual libraries (`python-pptx`, `python-docx`, `pdfplumber`, `openpyxl`) that keeps the install lean?

Also — go ahead and run those `ollama` commands when you get a chance and let me know what `ollama list` shows. Once you confirm the model is pulled and the test directory is ready, we start with Module 1.

<hr>
