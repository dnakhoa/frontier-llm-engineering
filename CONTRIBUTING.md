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
python3 tools/check_structure.py       # chapters, exercises, solutions, labs in sync
python3 tools/build_notebooks.py --check
python3 tools/run_labs.py              # or: run_labs.py lab04  for just one
```

All four run in CI. Running them locally first saves a round trip.

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
