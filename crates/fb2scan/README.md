# fb2scan

Parses [FB2](https://en.wikipedia.org/wiki/FictionBook) e-book files into a flat list of
plain-text sections suitable for feeding into an `SlmOracle` context.

Analogous to [`epubscan`](../epubscan) but for the FB2 format.

## Usage

```rust
use fb2scan::Fb2Scan;

let book = Fb2Scan::from_file(&"book.fb2".into())?;

for section in book.sections() {
    println!("# {}", section.title().unwrap_or("(no title)"));
    println!("{}", section.text());          // full section text
    println!("{:?}", section.language());    // auto-detected language, e.g. Some("English")

    for paragraph in section.pars() {       // individual paragraphs as &str slices
        println!("  - {}", paragraph);
    }
}
```

## API

### `Fb2Scan`

| Method | Description |
|---|---|
| `Fb2Scan::from_file(path)` | Parse an FB2 file from disk |
| `Fb2Scan::from_text(xml)` | Parse FB2 from an in-memory XML string |
| `scan.sections()` | Returns `&[Section]` — one entry per FB2 `<section>` with non-empty content |

### `Section`

| Method | Returns |
|---|---|
| `title()` | `Option<&str>` — section title, if present |
| `text()` | `&str` — full concatenated text of the section |
| `pars()` | `Vec<&str>` — individual paragraphs as sub-slices of `text()` |
| `language()` | `Option<&str>` — language name detected by `whatlang` (e.g. `"English"`) |

## Text rendering

FB2 inline markup is converted to Markdown-compatible plain text:

| FB2 element | Output |
|---|---|
| `<strong>` | `**bold**` |
| `<emphasis>` | `*italic*` |
| `<strikethrough>` | `~~text~~` |
| `<code>` | `` `code` `` |
| `<subtitle>` | `## subtitle` |
| `<poem>` / `<cite>` | Lines wrapped in `*...*` |
| Images, tables | Skipped |

Only main body sections are included; named bodies (notes, comments) are ignored.
