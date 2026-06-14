use super::*;
use std::io::Cursor;
use zip::ZipWriter;
use zip::write::SimpleFileOptions;

fn xhtml(body: &str) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>Test</title></head>
<body>{}</body>
</html>"#,
        body
    )
}

fn parse(body: &str) -> (String, Vec<(usize, usize)>) {
    parse_xhtml(&xhtml(body))
}

#[test]
fn plain_paragraph() {
    let (text, segs) = parse("<p>Hello world</p>");
    assert_eq!(segs.len(), 1);
    assert!(text.contains("Hello world"));
}

#[test]
fn bold_inline() {
    let (text, _) = parse("<p>A <strong>bold</strong> B</p>");
    assert!(text.contains("**bold**"), "got: {:?}", text);
}

#[test]
fn em_inline() {
    let (text, _) = parse("<p>A <em>em</em> B</p>");
    assert!(text.contains("*em*"), "got: {:?}", text);
}

#[test]
fn italic_tag() {
    let (text, _) = parse("<p>A <i>it</i> B</p>");
    assert!(text.contains("*it*"), "got: {:?}", text);
}

#[test]
fn strikethrough_inline() {
    let (text, _) = parse("<p>A <s>del</s> B</p>");
    assert!(text.contains("~~del~~"), "got: {:?}", text);
}

#[test]
fn code_inline() {
    let (text, _) = parse("<p>A <code>fn x()</code> B</p>");
    assert!(text.contains("`fn x()`"), "got: {:?}", text);
}

#[test]
fn heading_becomes_hash_prefix() {
    let (text, _) = parse("<h1>Chapter One</h1>");
    assert!(text.contains("## Chapter One"), "got: {:?}", text);
}

#[test]
fn h2_also_heading() {
    let (text, _) = parse("<h2>Section</h2>");
    assert!(text.contains("## Section"), "got: {:?}", text);
}

#[test]
fn image_skipped() {
    let (text, _) = parse("<p>Before</p><img src=\"cover.jpg\" alt=\"cover\"/><p>After</p>");
    assert!(!text.contains("cover"), "image leaked: {:?}", text);
    assert!(text.contains("Before"));
    assert!(text.contains("After"));
}

#[test]
fn script_content_skipped() {
    let (text, _) = parse("<p>Real</p><script>alert('x')</script>");
    assert!(!text.contains("alert"), "script leaked: {:?}", text);
    assert!(text.contains("Real"));
}

#[test]
fn style_content_skipped() {
    let (text, _) = parse("<style>body{color:red}</style><p>Text</p>");
    assert!(!text.contains("color"), "style leaked: {:?}", text);
    assert!(text.contains("Text"));
}

#[test]
fn multiple_paragraphs_become_segments() {
    let (_, segs) = parse("<p>One</p><p>Two</p><p>Three</p>");
    assert_eq!(segs.len(), 3);
}

#[test]
fn segment_content_by_index() {
    let (text, segs) = parse("<p>Alpha</p><p>Beta</p>");
    assert_eq!(segs.len(), 2);
    assert_eq!(text[segs[0].0..segs[0].1].trim(), "Alpha");
    assert_eq!(text[segs[1].0..segs[1].1].trim(), "Beta");
}

#[test]
fn empty_paragraph_not_a_segment() {
    let (_, segs) = parse("<p>A</p><p></p><p>B</p>");
    assert_eq!(segs.len(), 2);
}

fn make_epub(chapters: &[(&str, &str, &str)]) -> Vec<u8> {
    let buf = Cursor::new(Vec::new());
    let mut zip = ZipWriter::new(buf);
    let stored = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
    let deflated =
        SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("mimetype", stored).unwrap();
    zip.write_all(b"application/epub+zip").unwrap();

    let container = r#"<?xml version="1.0"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
  </rootfiles>
</container>"#;
    zip.start_file("META-INF/container.xml", deflated).unwrap();
    zip.write_all(container.as_bytes()).unwrap();

    let manifest_items: String = chapters
        .iter()
        .enumerate()
        .map(|(_i, (id, _, _))| {
            format!(
                r#"<item id="{id}" href="Text/{id}.xhtml" media-type="application/xhtml+xml"/>"#,
                id = id
            )
        })
        .collect::<Vec<_>>()
        .join("\n    ");
    let spine_items: String = chapters
        .iter()
        .map(|(id, _, _)| format!(r#"<itemref idref="{id}"/>"#, id = id))
        .collect::<Vec<_>>()
        .join("\n    ");
    let ncx_items: String = chapters
        .iter()
        .enumerate()
        .map(|(i, (id, title, _))| {
            format!(
                r#"<navPoint id="np{i}" playOrder="{po}">
      <navLabel><text>{title}</text></navLabel>
      <content src="Text/{id}.xhtml"/>
    </navPoint>"#,
                i = i,
                po = i + 1,
                title = title,
                id = id
            )
        })
        .collect::<Vec<_>>()
        .join("\n    ");

    let opf = format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>
    {manifest_items}
  </manifest>
  <spine toc="ncx">
    {spine_items}
  </spine>
</package>"#
    );
    zip.start_file("OEBPS/content.opf", deflated).unwrap();
    zip.write_all(opf.as_bytes()).unwrap();

    let ncx = format!(
        r#"<?xml version="1.0"?>
<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">
  <navMap>
    {ncx_items}
  </navMap>
</ncx>"#
    );
    zip.start_file("OEBPS/toc.ncx", deflated).unwrap();
    zip.write_all(ncx.as_bytes()).unwrap();

    for (id, _, body) in chapters {
        let content = format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml">
<head><title>{id}</title></head>
<body>{body}</body>
</html>"#,
            id = id,
            body = body
        );
        zip.start_file(format!("OEBPS/Text/{}.xhtml", id), deflated)
            .unwrap();
        zip.write_all(content.as_bytes()).unwrap();
    }

    zip.finish().unwrap().into_inner()
}

use std::io::Write;

fn scan_from_bytes(bytes: Vec<u8>) -> EpubScan {
    let cursor = Cursor::new(bytes);
    let mut archive = ZipArchive::new(cursor).expect("zip open");
    let sections = extract_sections(&mut archive).expect("extract");
    EpubScan { sections }
}

#[test]
fn single_chapter_extracted() {
    let epub = make_epub(&[("ch1", "Chapter One", "<p>Hello world</p>")]);
    let scan = scan_from_bytes(epub);
    assert_eq!(scan.sections().len(), 1);
    assert!(scan.sections()[0].text().contains("Hello world"));
}

#[test]
fn chapter_title_from_ncx() {
    let epub = make_epub(&[("ch1", "Chapter One", "<p>Content</p>")]);
    let scan = scan_from_bytes(epub);
    assert_eq!(scan.sections()[0].title(), Some("Chapter One"));
}

#[test]
fn multiple_chapters() {
    let epub = make_epub(&[
        ("ch1", "First", "<p>First content</p>"),
        ("ch2", "Second", "<p>Second content</p>"),
    ]);
    let scan = scan_from_bytes(epub);
    assert_eq!(scan.sections().len(), 2);
    assert_eq!(scan.sections()[0].title(), Some("First"));
    assert_eq!(scan.sections()[1].title(), Some("Second"));
}

#[test]
fn pars_count_matches_paragraphs() {
    let epub = make_epub(&[("ch1", "Ch", "<p>One</p><p>Two</p><p>Three</p>")]);
    let scan = scan_from_bytes(epub);
    assert_eq!(scan.sections()[0].pars().len(), 3);
}

#[test]
fn language_from_opf_metadata() {
    let epub = make_epub(&[("ch1", "Ch", "<p>Text</p>")]);
    let scan = scan_from_bytes(epub);
    assert_eq!(scan.sections()[0].language(), Some("en"));
}
