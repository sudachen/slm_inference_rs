# epubscan

Parses [EPUB](https://en.wikipedia.org/wiki/EPUB) e-book files into a flat list of
plain-text sections suitable for feeding into an `SlmOracle` context.

Analogous to [`fb2scan`](../fb2scan) but for the EPUB format.

## Usage

```rust
use epubscan::EpubScan;

let book = EpubScan::from_file(&"book.epub".into())?;

for section in book.sections() {
    println!("# {}", section.title().unwrap_or("(no title)"));
    println!("{}", section.text());          // full section text
    println!("{:?}", section.language());    // e.g. Some("English")

    for paragraph in section.pars() {       // individual paragraphs as &str slices
        println!("  - {}", paragraph);
    }
}
```

## API

### `EpubScan`

| Method | Description |
|---|---|
| `EpubScan::from_file(path)` | Open and parse an EPUB file from disk |
| `scan.sections()` | Returns `&[Section]` — one entry per spine item with non-empty content |

### `Section`

| Method | Returns |
|---|---|
| `title()` | `Option<&str>` — chapter title from the NCX table of contents, if present |
| `text()` | `&str` — full concatenated text of the section |
| `pars()` | `Vec<&str>` — individual paragraphs as sub-slices of `text()` |
| `language()` | `Option<&str>` — language from OPF metadata, or auto-detected by `whatlang` |

## How it works

1. Opens the EPUB ZIP archive and reads `META-INF/container.xml` to locate the OPF package document.
2. Parses the OPF to build a manifest (id → href/media-type) and the reading order spine.
3. Optionally parses the NCX navigation document to build a `filepath → title` map.
4. Iterates the spine in order, extracts each XHTML content document, and converts it to plain text.
5. Sections with empty text are skipped.

## XHTML text rendering

HTML markup inside content documents is converted to Markdown-compatible plain text:

| HTML element | Output |
|---|---|
| `<strong>`, `<b>` | `**bold**` |
| `<em>`, `<i>` | `*italic*` |
| `<s>`, `<del>`, `<strike>` | `~~text~~` |
| `<code>` | `` `code` `` |
| `<h1>`–`<h6>` | `## heading` |
| `<br>` | line break (`  \n`) |
| `<p>`, `<li>`, `<td>`, `<blockquote>`, block containers | paragraph boundary |
| `<script>`, `<style>`, `<svg>` | skipped entirely |
| Everything outside `<body>` | ignored |
