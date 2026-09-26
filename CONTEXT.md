# Frontier LLM Engineering

An open book, with exercises and runnable labs, about how frontier labs build large language models.
This glossary fixes the words we use when we talk about keeping the book accurate and current.

## Claims and sources

**Primary source**:
A lab's own technical report, paper, model card, official blog post, or published artifact (config file, tokenizer file, repository).
_Avoid_: source (unqualified), reference

**Never-true claim**:
A statement about what a primary source says or contains that the source contradicts, and so was wrong on the day it was written. A place where the book contradicts itself also counts.
_Avoid_: error (unqualified), mistake, outdated claim

**Stale claim**:
A statement that was accurate against its sources when written but has since been superseded by newer work.
_Avoid_: wrong claim, outdated fact

**Unsupported claim**:
A statement presented as fact with no primary source behind it, whether or not it happens to be true.
_Avoid_: guess, estimate (unless it is labelled as one)

**Pinpoint citation**:
A citation precise enough to check in a minute: the primary source's version plus section and table or figure, or a file path plus commit for published artifacts.
_Avoid_: reference (unqualified)

**Fact sheet**:
An appendix page for one model that records every fact the book relies on about it, each fact with a pinpoint citation to its primary source.
_Avoid_: spec sheet, model card (that term belongs to the lab's document)

**Fact**:
One row of a fact sheet: a single value or statement with its pinpoint citation. Chapters link to it rather than restating it.

**Real config**:
A config or tokenizer block presented as a specific model's actual configuration, generated from the published file and labelled with its origin.
_Avoid_: example config

**Illustrative config**:
A hand-written config block that shows a shape or idea and is labelled as not belonging to any real model.

## Releases

**Correction release**:
A release that fixes only never-true claims, shipped on its own, before and separately from any refresh.
_Avoid_: bugfix release, patch

**Refresh**:
Work that replaces stale claims and fills gaps so a chapter reflects its field as of a stated date.
_Avoid_: update, rewrite

**Citation check**:
An adversarial pass that verifies every claim a chapter makes about a primary source, against that source, before the chapter's refresh merges.
_Avoid_: review, fact-check (unqualified)

**Errata page**:
The permanent appendix page that lists every never-true claim the book has published, what the source actually says, and when it was fixed.

**Currency stamp**:
The "Current as of <month year>" line on each chapter, naming the month its claims were last checked against the field. Only a refresh moves it; corrections and typo fixes do not.
_Avoid_: last updated, revision date

## Chapters

**Case study**:
A chapter that reads one lab's report end to end (currently DeepSeek-V3 and DeepSeek-R1), closed by a "what came next" section tracing that model's successors.
