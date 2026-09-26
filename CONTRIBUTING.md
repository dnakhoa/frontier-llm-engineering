# Contributing

This book exists to be free and correct. Both parts need help.

## The contributions that matter most

In rough order of value:

1. **A citation we got wrong.** Wrong author, wrong year, wrong arXiv number, or — worst — a claim attributed to a paper that does not actually make it. Open an issue with the chapter, the section number, and the correct source.
2. **A number that has been superseded.** The frontier moves; a technical report drops and a number in Chapter 5 is now historical. We would rather say "as of the V3 report, X; the V4 report revised this to Y" than quietly carry a stale figure.
3. **A lab that no longer runs.** Libraries change APIs. If a lab fails on current versions, that is a bug, and CI should have caught it — tell us what broke and on which versions.
4. **An exercise whose solution is wrong.** Or an exercise that is unsolvable from the chapter alone, which is the same bug wearing a different hat.
5. **A chapter that is unclear.** Vaguer, harder to act on, but real. Say which paragraph lost you and what you thought it meant.

## Ground rules for prose

Read [the style guide](book/appendix/style-guide.md) first — it is short and it is enforced in review.

The three rules that get violated most:

- **Say what is known and say what is not.** "The exact data mix has not been published" is a valid and useful sentence. Inventing a plausible number is not.
- **No marketing language.** No "revolutionary," "groundbreaking," "game-changing" unless a primary source used the word and you are quoting it.
- **Cite real things.** Every `[N]` must resolve to a real entry in [the reference list](book/appendix/b-references.md), which must resolve to a real paper, report, or repository.

## Claims about sources: fact sheets and the citation check

The book's worst failures have not been typos. They have been confident sentences about what a report says that the report does not say: a parallelism layout, a block-quoted sentence, a config file. The book uses three words for the ways a claim can go wrong:

- A **never-true claim** contradicts its primary source (or the book itself) and was wrong on the day it was written. Fixing one earns an entry on the [errata page](book/appendix/d-errata.md).
- A **stale claim** was accurate when written and has been overtaken by newer work. It goes in the CHANGELOG, and the chapter's refresh moves its currency stamp.
- An **unsupported claim** is stated as fact with no source behind it. Source it, or label it as an estimate.

Three rules follow from that.

1. **Facts about a specific model live in its [fact sheet](book/appendix/c-fact-sheets.md).** Chapters, exercises, solutions and labs link to the row rather than restating it. If a calculation needs the value inline, give it with a link to the row. A new fact needs a pinpoint citation: the source's version plus section, table or figure, or a file path plus commit for published files.
2. **A config block is either real or illustrative, and says which.** A block presented as a real model's configuration is generated from the published file (see the style guide). Everything else carries an `ILLUSTRATIVE` label.
3. **Every change that adds or changes a claim about a primary source passes a citation check before it merges.** It is part of the pull-request checklist:

   - [ ] **Citation check.** For every claim this PR makes about what a source says or contains: I opened the cited version of the source, found the section, table, figure or file line, and confirmed that it says this. Values we derived are marked *our arithmetic* and show their inputs.

   The check is adversarial. The question is "where would this be wrong?", not "does this look plausible?". Checking the claim against another secondary source does not count, and neither does checking it against another chapter of this book. Four chapters once agreed with each other about a layout the source contradicts. An agent can run the check, but it must quote the passage it checked against. CI cannot run it: `tools/check_structure.py` confirms that a citation *exists*, never that it is *right*.

## Ground rules for labs

A lab earns its place by making a mechanism concrete that prose cannot. It is not a tutorial and it is not a benchmark.

Hard requirements, all enforced by CI:

- **Runs on a free Colab T4, and also on CPU.** Detect the device; do not assume CUDA.
- **Runs when copy-pasted into a single Colab cell.** No local file reads, no relative imports between labs, no `argparse` that breaks without a terminal.
- **Honours `FLE_SMOKE_TEST=1`** by shrinking every loop, dataset, and model to the smallest size that still exercises the code path. CI sets this; a lab that ignores it will time out.
- **Finishes in under 15 minutes** on a T4 at full size, and under a minute in smoke-test mode.
- **Is authored as `.py` with `# %%` cell markers**, not as a checked-in `.ipynb`. Notebooks are generated:

  ```bash
  python3 tools/build_notebooks.py
  ```

  Commit both the `.py` and the regenerated `.ipynb`.

- **Is referenced from at least one chapter or exercise set.** Orphaned labs fail `tools/check_structure.py`.

## Before you open a pull request

```bash
python3 tools/check_links.py           # every internal link and anchor resolves
python3 -m unittest discover -s tools -p "test_*.py"   # the checkers themselves
python3 tools/check_structure.py       # chapters, exercises, solutions, labs, stamps, errata, fact sheets
python3 tools/build_notebooks.py --check
python3 tools/run_labs.py              # or: run_labs.py lab04  for just one
```

All five run in CI. None of them replaces the citation check above. Running them locally first saves a round trip.

## Adding a chapter

Chapters are numbered and their filenames encode the number: `book/part-N-slug/NN-chapter-slug.md`. A new chapter needs, at minimum:

- The chapter file, following the six-part structure in the style guide (header block, motivating case study, numbered sections, "JD, decoded", takeaways, references).
- An entry in [`SUMMARY.md`](SUMMARY.md).
- `exercises/chNN.md` and `exercises/solutions/chNN.md`. `tools/check_structure.py` will fail without them.
- Any new citations added to [the reference list](book/appendix/b-references.md) in numeric order.

## Repository URL

The Colab badges in generated notebooks and the CI badge in the README point at a
GitHub org/repo pair that is set in exactly two places:

- `GITHUB_REPO` and `BRANCH` in [`tools/build_notebooks.py`](tools/build_notebooks.py)
- the badge URLs at the top of [`README.md`](README.md)

If you fork this, change both, then re-run `tools/build_notebooks.py` so the Colab
links point at your fork.

## Code of conduct

Participation is governed by [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). It is short: be decent, argue about the work and not the person.

## Licensing your contribution

By contributing you agree that your prose is licensed under [CC BY-SA 4.0](LICENSE) and your code under [MIT](LICENSE-CODE), matching the rest of the project. If you cannot agree to that, please open an issue rather than a pull request and we will figure something out.
