use std::collections::HashMap;
use std::io::{BufReader, Read};
use std::path::PathBuf;
use anyhow::{Context as _, Result};
use quick_xml::{Decoder, Reader};
use quick_xml::events::{BytesStart, Event};
use zip::ZipArchive;

pub struct EpubScan {
    sections: Vec<Section>,
}

impl EpubScan {
    pub fn from_file(path: &PathBuf) -> Result<Self> {
        let file = std::fs::File::open(path).context("Cannot open EPUB file")?;
        let mut archive =
            ZipArchive::new(BufReader::new(file)).context("Failed to open EPUB as ZIP")?;
        let sections = extract_sections(&mut archive)?;
        Ok(Self { sections })
    }

    pub fn sections(&self) -> &[Section] {
        &self.sections
    }
}

pub struct Section {
    title: Option<String>,
    text: String,
    language: Option<String>,
    segment_indices: Vec<(usize, usize)>,
}

impl Section {
    pub fn title(&self) -> Option<&str> {
        self.title.as_deref()
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn pars(&self) -> Vec<&str> {
        self.segment_indices
            .iter()
            .map(|&(s, e)| &self.text[s..e])
            .collect()
    }

    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }
}

fn read_zip_entry<R: Read + std::io::Seek>(archive: &mut ZipArchive<R>, name: &str) -> Result<String> {
    let mut entry = archive
        .by_name(name)
        .with_context(|| format!("Entry not found in EPUB: {}", name))?;
    let mut bytes = Vec::new();
    entry.read_to_end(&mut bytes)?;
    Ok(String::from_utf8_lossy(&bytes).into_owned())
}

fn dir_of(path: &str) -> &str {
    match path.rfind('/') {
        Some(i) => &path[..i],
        None => "",
    }
}

fn normalize_path(path: &str) -> String {
    let mut parts: Vec<&str> = Vec::new();
    for component in path.split('/') {
        match component {
            ".." => { parts.pop(); }
            "." | "" => {}
            s => parts.push(s),
        }
    }
    parts.join("/")
}

fn resolve(base_dir: &str, href: &str) -> String {
    let href = href.split('#').next().unwrap_or(href);
    let combined = if base_dir.is_empty() {
        href.to_string()
    } else {
        format!("{}/{}", base_dir, href)
    };
    normalize_path(&combined)
}

fn find_opf_path<R: Read + std::io::Seek>(archive: &mut ZipArchive<R>) -> Result<String> {
    let xml = read_zip_entry(archive, "META-INF/container.xml")?;
    let mut reader = Reader::from_str(&xml);
    let decoder = reader.decoder();
    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Empty(e)) | Ok(Event::Start(e)) => {
                if e.name().local_name().as_ref() == b"rootfile" {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"full-path" {
                            return Ok(attr.decode_and_unescape_value(decoder)?.into_owned());
                        }
                    }
                }
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }
    anyhow::bail!("OPF path not found in container.xml")
}

struct ManifestItem {
    href: String,
    media_type: String,
}

struct OpfData {
    opf_dir: String,
    manifest: HashMap<String, ManifestItem>,
    spine: Vec<String>,
    ncx_id: Option<String>,
    language: Option<String>,
}

fn parse_opf<R: Read + std::io::Seek>(archive: &mut ZipArchive<R>, opf_path: &str) -> Result<OpfData> {
    let xml = read_zip_entry(archive, opf_path)?;
    let opf_dir = dir_of(opf_path).to_string();
    let mut manifest: HashMap<String, ManifestItem> = HashMap::new();
    let mut spine: Vec<String> = Vec::new();
    let mut ncx_id: Option<String> = None;
    let mut language: Option<String> = None;
    let mut in_metadata = false;
    let mut in_language = false;

    let mut reader = Reader::from_str(&xml);
    let decoder = reader.decoder();
    let mut buf = Vec::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                match e.name().local_name().as_ref() {
                    b"metadata" => in_metadata = true,
                    b"language" if in_metadata => in_language = true,
                    b"spine" => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"toc" {
                                ncx_id = Some(attr.decode_and_unescape_value(decoder).unwrap_or_default().into_owned());
                            }
                        }
                    }
                    b"item" => parse_manifest_item(&e, decoder, &mut manifest),
                    b"itemref" => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"idref" {
                                spine.push(attr.decode_and_unescape_value(decoder).unwrap_or_default().into_owned());
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                match e.name().local_name().as_ref() {
                    b"item" => parse_manifest_item(&e, decoder, &mut manifest),
                    b"itemref" => {
                        for attr in e.attributes().flatten() {
                            if attr.key.as_ref() == b"idref" {
                                spine.push(attr.decode_and_unescape_value(decoder).unwrap_or_default().into_owned());
                            }
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::End(e)) => match e.name().local_name().as_ref() {
                b"metadata" => in_metadata = false,
                b"language" => in_language = false,
                _ => {}
            },
            Ok(Event::Text(t)) if in_language && language.is_none() => {
                let s = t.xml10_content().unwrap_or_default().trim().to_string();
                if !s.is_empty() { language = Some(s); }
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }

    if ncx_id.is_none() {
        for (id, item) in &manifest {
            if item.media_type == "application/x-dtbncx+xml" {
                ncx_id = Some(id.clone());
                break;
            }
        }
    }

    Ok(OpfData { opf_dir, manifest, spine, ncx_id, language })
}

fn parse_manifest_item(e: &BytesStart, decoder: Decoder, manifest: &mut HashMap<String, ManifestItem>) {
    let mut id = String::new();
    let mut href = String::new();
    let mut media_type = String::new();
    for attr in e.attributes().flatten() {
        match attr.key.as_ref() {
            b"id" => id = attr.decode_and_unescape_value(decoder).unwrap_or_default().into_owned(),
            b"href" => href = attr.decode_and_unescape_value(decoder).unwrap_or_default().into_owned(),
            b"media-type" => media_type = attr.decode_and_unescape_value(decoder).unwrap_or_default().into_owned(),
            _ => {}
        }
    }
    if !id.is_empty() && !href.is_empty() {
        manifest.insert(id, ManifestItem { href, media_type });
    }
}

fn parse_ncx<R: Read + std::io::Seek>(
    archive: &mut ZipArchive<R>,
    ncx_zip_path: &str,
) -> Result<HashMap<String, String>> {
    let ncx_dir = dir_of(ncx_zip_path).to_string();
    let xml = read_zip_entry(archive, ncx_zip_path)?;
    let mut toc: HashMap<String, String> = HashMap::new();
    let mut reader = Reader::from_str(&xml);
    let decoder = reader.decoder();
    let mut buf = Vec::new();
    let mut in_nav_label = false;
    let mut in_text = false;
    let mut current_label = String::new();
    let mut current_src = String::new();

    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => match e.name().local_name().as_ref() {
                b"navLabel" => {
                    in_nav_label = true;
                    current_label.clear();
                }
                b"text" if in_nav_label => { in_text = true; }
                b"content" => {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"src" {
                            current_src = resolve(&ncx_dir, &attr.decode_and_unescape_value(decoder).unwrap_or_default());
                        }
                    }
                }
                _ => {}
            },
            Ok(Event::Empty(e)) => {
                if e.name().local_name().as_ref() == b"content" {
                    for attr in e.attributes().flatten() {
                        if attr.key.as_ref() == b"src" {
                            current_src = resolve(&ncx_dir, &attr.decode_and_unescape_value(decoder).unwrap_or_default());
                        }
                    }
                }
            }
            Ok(Event::End(e)) => match e.name().local_name().as_ref() {
                b"navLabel" => {
                    in_nav_label = false;
                    in_text = false;
                }
                b"text" => { in_text = false; }
                b"navPoint" => {
                    if !current_src.is_empty() && !current_label.is_empty() {
                        toc.entry(current_src.clone())
                            .or_insert_with(|| current_label.trim().to_string());
                    }
                    current_src.clear();
                    current_label.clear();
                }
                _ => {}
            },
            Ok(Event::Text(t)) if in_text => {
                current_label.push_str(&t.xml10_content().unwrap_or_default());
            }
            Ok(Event::Eof) => break,
            _ => {}
        }
        buf.clear();
    }

    Ok(toc)
}

fn append_segment(text: &mut String, indices: &mut Vec<(usize, usize)>, content: &str) {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return;
    }
    let start = text.len();
    text.push_str(trimmed);
    text.push_str("\n\n");
    indices.push((start, text.len()));
}

enum InlineMarker {
    Bold,
    Italic,
    Strike,
    Code,
}

impl InlineMarker {
    fn marker(&self) -> &'static str {
        match self {
            Self::Bold => "**",
            Self::Italic => "*",
            Self::Strike => "~~",
            Self::Code => "`",
        }
    }
}

pub(crate) fn parse_xhtml(xml: &str) -> (String, Vec<(usize, usize)>) {
    let mut reader = Reader::from_str(xml);
    reader.config_mut().check_end_names = false;

    let mut text = String::new();
    let mut indices: Vec<(usize, usize)> = Vec::new();
    let mut current = String::new();
    let mut inline_stack: Vec<InlineMarker> = Vec::new();
    let mut skip_depth: usize = 0;
    let mut in_body = false;

    let mut buf = Vec::new();
    loop {
        match reader.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                let tag = e.name().local_name();
                let tag = tag.as_ref();

                if tag == b"body" {
                    in_body = true;
                    continue;
                }
                if !in_body { continue; }

                if skip_depth > 0 {
                    skip_depth += 1;
                    continue;
                }

                match tag {
                    b"script" | b"style" | b"svg" => { skip_depth = 1; }
                    b"strong" | b"b" => {
                        current.push_str("**");
                        inline_stack.push(InlineMarker::Bold);
                    }
                    b"em" | b"i" => {
                        current.push('*');
                        inline_stack.push(InlineMarker::Italic);
                    }
                    b"s" | b"del" | b"strike" => {
                        current.push_str("~~");
                        inline_stack.push(InlineMarker::Strike);
                    }
                    b"code" => {
                        current.push('`');
                        inline_stack.push(InlineMarker::Code);
                    }
                    _ => {}
                }
            }
            Ok(Event::Empty(e)) => {
                let tag = e.name().local_name();
                if in_body && skip_depth == 0 && tag.as_ref() == b"br" {
                    current.push_str("  \n");
                }
            }
            Ok(Event::End(e)) => {
                let tag = e.name().local_name();
                let tag = tag.as_ref();

                if tag == b"body" {
                    append_segment(&mut text, &mut indices, &current);
                    current.clear();
                    in_body = false;
                    continue;
                }
                if !in_body { continue; }

                if skip_depth > 0 {
                    skip_depth -= 1;
                    continue;
                }

                match tag {
                    b"strong" | b"b" | b"em" | b"i" | b"s" | b"del" | b"strike" | b"code" => {
                        if let Some(m) = inline_stack.pop() {
                            current.push_str(m.marker());
                        }
                    }
                    b"p" | b"li" | b"dt" | b"dd" | b"th" | b"td" | b"blockquote" => {
                        append_segment(&mut text, &mut indices, &current);
                        current.clear();
                        inline_stack.clear();
                    }
                    b"div" | b"section" | b"article" | b"header" | b"footer" | b"main" | b"nav" => {
                        append_segment(&mut text, &mut indices, &current);
                        current.clear();
                        inline_stack.clear();
                    }
                    b"h1" | b"h2" | b"h3" | b"h4" | b"h5" | b"h6" => {
                        let trimmed = current.trim().to_string();
                        current.clear();
                        inline_stack.clear();
                        if !trimmed.is_empty() {
                            append_segment(&mut text, &mut indices, &format!("## {}", trimmed));
                        }
                    }
                    _ => {}
                }
            }
            Ok(Event::Text(t)) => {
                if in_body && skip_depth == 0 {
                    current.push_str(&t.xml10_content().unwrap_or_default());
                }
            }
            Ok(Event::Eof) | Err(_) => break,
            _ => {}
        }
        buf.clear();
    }

    (text, indices)
}

fn extract_sections<R: Read + std::io::Seek>(archive: &mut ZipArchive<R>) -> Result<Vec<Section>> {
    let opf_path = find_opf_path(archive)?;
    let opf = parse_opf(archive, &opf_path)?;

    let mut toc_map: HashMap<String, String> = HashMap::new();
    if let Some(ncx_id) = &opf.ncx_id {
        if let Some(item) = opf.manifest.get(ncx_id) {
            let ncx_path = resolve(&opf.opf_dir, &item.href);
            if let Ok(map) = parse_ncx(archive, &ncx_path) {
                toc_map = map;
            }
        }
    }

    let mut sections = Vec::new();

    for idref in &opf.spine {
        let item = match opf.manifest.get(idref) {
            Some(i) => i,
            None => continue,
        };
        if !item.media_type.contains("xhtml") && !item.media_type.contains("html") {
            continue;
        }
        let zip_path = resolve(&opf.opf_dir, &item.href);
        let xhtml = match read_zip_entry(archive, &zip_path) {
            Ok(s) => s,
            Err(_) => continue,
        };

        let (text, segment_indices) = parse_xhtml(&xhtml);
        if text.is_empty() {
            continue;
        }

        let title = toc_map.get(&zip_path).cloned();
        let language = opf.language.clone().or_else(|| {
            whatlang::detect(&text).map(|x| x.lang().eng_name().to_string())
        });

        sections.push(Section { title, text, language, segment_indices });
    }

    Ok(sections)
}

#[cfg(test)]
mod tests;
