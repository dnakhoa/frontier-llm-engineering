# Fact sheets

A fact sheet records every fact the book relies on about one model: its layout, layer count, data mix, schedule, precision and cost. Each fact carries a **pinpoint citation**: the source's version plus a section, table or figure, or a file path plus commit for published files. A reader should be able to check any row in about a minute.

Chapters, exercises, solutions and labs **link** to a fact rather than restating it. Where a calculation needs the value inline, the value appears with a link to its row. This rule exists because of a real failure. The book once stated the same wrong DeepSeek-V3 parallelism layout in four chapters, and all four copies agreed with each other. A check that looks for disagreement between copies would never have caught it. With one cited copy, there is only one place to get wrong and one place to check. (The decision is recorded in `docs/adr/0002-fact-sheets-single-source.md` in the repository.)

## The sheets

| Model | Primary source |
|---|---|
| [DeepSeek-V3](fact-sheets/deepseek-v3.md) | DeepSeek-V3 Technical Report, arXiv:2412.19437v2 |

More sheets are added as the corrections and refreshes need them.

## What a fact sheet does not do

A fact sheet says what a source **reports**. It does not say whether the source is right, and it does not say what is current. A fact is never "outdated": it says what the V3 report says, and that stays true. Whether V3 is still the frontier is a question for the chapter and its [currency stamp](../how-to-use-this-book.md#reading-the-chapters).

Rows marked *our arithmetic* are values we derived from reported numbers, with the derivation shown. They are ours, not the lab's.

`tools/check_structure.py` fails CI when a row has no pinpoint citation. It cannot tell whether the citation is *correct*. That is the **citation check**, in which a person or an agent reads the source itself before a change merges; see CONTRIBUTING.md. Found a row that disagrees with its source? That is the most valuable issue you can file.
