# LLM course data

The module does not hide downloads behind a dataset library. Every source,
revision, license note, and known checksum is declared in `sources.json`, and the
standard-library downloader writes a second manifest with resolved URLs, byte
counts, and observed SHA-256 hashes.

```bash
uv run --extra llm python -m experiments.llm.download_data --dataset books
uv run --extra llm python -m experiments.llm.download_data --dataset sft
uv run --extra llm python -m experiments.llm.download_data --dataset preference
uv run --extra llm python -m experiments.llm.prepare_data --tokenizer bpe
```

The default Project Gutenberg collection is a random sample of 16 English text
records from the official weekly catalog. About 10% of the selected works (two
at the default size) are randomly assigned to validation as whole books. This
avoids the severe leakage caused by splitting adjacent windows from one book.

## Random, incremental pre-training downloads

`--book-limit N` is the total active book count. N may be any integer from 2 up
to the eligible population in the current [official weekly CSV
catalog](https://www.gutenberg.org/ebooks/offline_catalogs.html); there is no
hard-coded 16- or 100-book ceiling. For example:

```bash
make download-llm-data LLM_BOOK_LIMIT=64
make prepare-llm-data \
  LLM_VOCAB_SIZE=1024 \
  LLM_TOKENIZER_CHARACTERS=5000000
```

The equivalent direct commands are:

```bash
uv run --extra llm python -m experiments.llm.download_data \
  --dataset books --book-limit 64 --request-delay 2
uv run --extra llm python -m experiments.llm.prepare_data \
  --tokenizer bpe --vocab-size 1024 --tokenizer-characters 5000000
```

Each run filters the catalog to English `Text` records, removes the homework
title and normalized-title duplicates, then samples without replacement. It is
cache-aware: eligible nonempty files already named `raw/books/pg<ID>.txt` are
randomized and used before missing candidates. Thus a run that grows a four-book
cache to seven books reuses four files and downloads only three. A repeated or
smaller run may choose a different active subset, but it does not delete cached
files or request any selected file that already exists.

By default a fresh 64-bit random seed is generated on every run. The seed,
catalog SHA-256, exact selected titles/splits, cache/download counts, resolved
URLs, and hashes are written to `download_manifest.json`; preprocessing reads
that manifest. Use `--selection-seed 2026` (or
`LLM_SELECTION_SEED=2026` with `make`) for a controlled classroom run. The
catalog itself is cached at `raw/catalog/pg_catalog.csv.gz`; add
`--refresh-catalog` only when you want a newer snapshot.

Book payloads come from generated UTF-8 text files on `gutenberg.pglaf.org`,
with its main-collection ZIP files as fallbacks. Project
Gutenberg's [registered mirror
list](https://www.gutenberg.org/MIRRORS.ALL) identifies as its high-speed San
Diego mirror. The fallback archive path is derived from the public main-collection layout
described in the [mirroring
guide](https://www.gutenberg.org/help/mirroring.html), with two seconds between
new downloads. This avoids automated bulk requests to the human-facing website,
which Gutenberg's [terms of
use](https://www.gutenberg.org/policy/terms_of_use.html) prohibit. For very
large corpora, use a private/authorized mirror and pass its ZIP or plain-text
template with `--book-url-template`; placeholders `{id}` and `{id_path}` are
supported. Gutenberg's [robot harvest
route](https://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en) and
weekly all-text archive are alternatives for bulk acquisition.

More data does not automatically increase training compute. After preparation,
check `processed/metadata.json` for the new training-token count. The default MPS
preset still processes about 24.6 million token positions (6,000 updates times
batch 16 times context 256), so a 64-book corpus may receive less than one
effective pass. Time a short run, then set the step/token budget intentionally.

The post-training labs use Databricks Dolly 15k for SFT and Anthropic HH-RLHF
`harmless-base` for pairwise reward learning. HH-RLHF contains offensive and
unsafe text by design; do not project raw examples in class.

Downloaded and processed files are intentionally ignored by Git because the
entire corpus is reproducible from the manifest. Project Gutenberg states that
the vast majority of its books are public domain in the United States and tells
non-U.S. users to check local copyright law. Each eBook's embedded license remains
the authoritative item-level notice.
