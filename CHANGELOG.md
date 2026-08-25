# Changelog

The frontier moves; this file records when the book moved with it.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- **The site is now built with mdBook** (the same toolchain and rust/navy theme as Everything Data Structures), replacing Honkit. All published URLs are unchanged. Math is rendered client-side with KaTeX using Pandoc-style `$` delimiters — inline math must hug its dollar signs, which is what keeps prose like "costs $2 and $15" from being parsed as math. Mermaid diagrams render client-side and follow the light/dark theme.

### Added
- **Chapters 11-26** — Parts III (post-training), IV (infrastructure), and V (the job), written from scratch. ~90,000 words across 16 chapters.
- **All 26 exercise sets and 26 worked solution sets** (~125,000 words): arithmetic drills, design problems, paper-reading prompts, and lab experiments with stated predictions.
- **Five runnable labs**, all CPU-only and all tested in CI: BPE tokenization and multilingual fertility (Ch 4), MoE routing and load imbalance (Ch 5), a distributed-training memory and throughput model (Ch 6), a collective-communication cost model (Ch 7), and SFT loss masking and sequence packing (Ch 11).
- References 51-96, covering the post-training, infrastructure, and evaluation literature.
- Open-repository scaffolding: dual license (CC BY-SA 4.0 for prose, MIT for code), contribution guide, code of conduct.
- GitBook Git Sync configuration (`.gitbook.yaml`) and Honkit config (`book.json`) so the repo renders as a book without any further setup.
- `tools/check_links.py` — validates every internal link *and* every heading anchor across the book.
- `tools/check_structure.py` — enforces that every chapter has exercises, every exercise set has solutions, and no lab is orphaned.
- `tools/build_notebooks.py` — generates Colab-ready `.ipynb` files from the `.py` labs.
- `tools/run_labs.py` — runs every lab; CI runs it on a CPU-only runner on each commit.
- CI workflow running all four checks.

### Known gaps
- **The chapters name roughly fourteen labs that do not exist yet.** Chapters 3, 8, 9, 12, 13, 15, 16, 20, 21, 22, and 23 each describe a lab in their closing section; those are specifications, not links. Nothing links to a missing file, and [`labs/README.md`](labs/README.md) lists exactly which are outstanding.

### Fixed
- **232 broken reference links.** Every chapter linked to `appendix/b-references.md`, which resolves relative to the chapter's own directory and therefore pointed at nothing. Corrected to `../appendix/b-references.md`.
- **32 broken heading anchors**, including 23 instances of `#5-llama3` (the heading slug is `#5-llama-3`), plus `#50-dualpipe`, `#8-dedup`, and `#48-ntk-aware-scaling`.
- Reference 44 (`µTransfer`) and reference 29 (`PagedAttention and vLLM (see #23)`) had headings whose slugs contained a non-ASCII character and a `#`, making their anchors unlinkable. Renamed to ASCII-only headings.
- Removed a stray build report (`06-deliverable.md`) that was sitting inside the book tree and colliding with Chapter 6's number. It now lives in `.meta/build-reports/`.

## [0.1.0]

### Added
- Preface, Part I (Chapters 1–2), Part II (Chapters 3–10).
- Glossary and a 50-entry reference list.
- Style guide.
