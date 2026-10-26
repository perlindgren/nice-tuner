//! A hand-written RON subset: the parser, tree model and writer shared
//! by the `ron_view` viewer and `sprite_util`'s sidecar panels.
//!
//! The subset speaks line and (nestable) block comments, `!` type
//! markers, `Name(..)` and `Name {..}` structs, named and positional
//! fields, `(..)` anonymous structs, arrays, maps with string-ish keys,
//! strings and chars (kept verbatim, escapes and all), integers,
//! floats, exponents, underscores, bools, and enum paths — unit
//! (`Kind::Bee`), newtype (`Kind::Bug(Count(3))`) and struct variants
//! (`Watering::PingPong { .. }`) — with trailing commas allowed
//! everywhere. It errors with a line number rather than guessing. Raw
//! (`r".."`) and multi-line strings are not in the subset.
//!
//! This is not a deserializer: the module owns no types it decodes into
//! and asks no type what to expect. It is lossless by preservation —
//! comments, unknown keys, numbers and strings all stay text, riding in
//! the tree exactly as the source wrote them — so a file that is read,
//! edited and saved returns everything the program did not recognize.
//! The one deliberate loss is the `!Type(..)` marker: the parser
//! consumes it and keeps only the value inside, because the marker
//! names the writer's type, which neither a viewer nor an editor needs.
//!
//! The tree carries the source's own shapes: containers hold their fold
//! state in `open`, and every value travels in an [`Item`] with the
//! comments that surrounded it, which is what makes the round trip
//! lossless. What the module does not own is the *view*: the flattener
//! that turns a tree into drawable rows (`VKind`, `Row`, `flatten`,
//! `layout`) belongs to whoever paints it and lives in
//! `examples/ron_view/view.rs`, path-included by both examples.

/// What an atom's text is, for coloring and weighting it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    /// `true` or `false`.
    Bool,
    /// An integer or float, kept verbatim: digits, `_` separators,
    /// fraction, exponent and type suffix all ride along as text.
    Num,
    /// A `"quoted"` string, quotes and escapes included.
    Str,
    /// A `'quoted'` char, verbatim like a string.
    Char,
    /// A path-like atom: an enum unit variant, a bare identifier.
    Path,
}

/// A parsed RON value. Containers carry their fold state: the view
/// mutates these flags in place, and flattens to rows from them.
/// Container children are [`Item`]s, so every value travels with the
/// comments that surrounded it in the source — reordering, deleting or
/// editing a value carries its comments along, and saving writes them
/// back. `tail` holds the comments that sat between the last entry and
/// the container's closing bracket.
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    /// Text, verbatim from the source (quotes and escapes included).
    Atom(String, Kind),
    /// `[..]`.
    Seq {
        open: bool,
        items: Vec<Item>,
        tail: String,
    },
    /// `Name(..)`, `(..)`, `Name {..}` or `{..}` with named fields;
    /// field names are empty for positional (tuple) entries.
    Struct {
        open: bool,
        head: String,
        curly: bool,
        fields: Vec<(String, Item)>,
        tail: String,
    },
    /// `{ key: value, .. }` — keys rendered to display strings.
    Map {
        open: bool,
        entries: Vec<(String, Item)>,
        tail: String,
    },
}

/// One entry of a container: the value plus its comments, verbatim.
/// `lead` is the run of whole-line comments above the entry (one per
/// line, `\n` between); `trail` is the comment that followed it on
/// the same line. Newly added entries carry neither.
#[derive(Debug, Clone, PartialEq)]
pub struct Item {
    pub lead: String,
    pub trail: String,
    /// The entry's value.
    pub val: Val,
}

impl Item {
    /// A bare value, no comments — the shape `sprite_util` appends.
    pub fn plain(val: Val) -> Self {
        Self {
            lead: String::new(),
            trail: String::new(),
            val,
        }
    }
}

impl Val {
    /// The value's one-line text: atoms verbatim, containers as a
    /// bracketed ellipsis — enough for map keys, which the parser
    /// stringifies through it.
    pub fn display(&self) -> String {
        match self {
            Val::Atom(s, _) => s.clone(),
            Val::Seq { .. } => "[…]".into(),
            Val::Struct { head, curly, .. } => {
                format!(
                    "{head}{}…{}",
                    if *curly { " {" } else { "(" },
                    if *curly { "}" } else { ")" }
                )
            }
            Val::Map { .. } => "{…}".into(),
        }
    }

    /// One line for a folded container: the whole subtree inline when
    /// it fits `INLINE_MAX` characters — a flower anchor reads
    /// `(317.0, 671.0)`, not `(…)` — and [`Val::display`]'s bracketed
    /// ellipsis when it doesn't.
    pub fn preview(&self) -> String {
        let one = self.inline();
        if one.chars().count() <= INLINE_MAX {
            one
        } else {
            self.display()
        }
    }

    /// The value as one full line of RON — every child, no breaks.
    /// [`Val::preview`] is the caller that caps it.
    pub fn inline(&self) -> String {
        match self {
            Val::Atom(s, _) => s.clone(),
            Val::Seq { items, .. } => format!(
                "[{}]",
                items
                    .iter()
                    .map(|it| it.val.inline())
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Val::Struct {
                head,
                curly,
                fields,
                ..
            } => {
                let (o, c) = if *curly {
                    if head.is_empty() {
                        ("{", "}")
                    } else {
                        (" { ", " }")
                    }
                } else {
                    ("(", ")")
                };
                let inner = fields
                    .iter()
                    .map(|(k, it)| {
                        if k.is_empty() {
                            it.val.inline()
                        } else {
                            format!("{k}: {}", it.val.inline())
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("{head}{o}{inner}{c}")
            }
            Val::Map { entries, .. } => format!(
                "{{{}}}",
                entries
                    .iter()
                    .map(|(k, it)| format!("{k}: {}", it.val.inline()))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }

    /// The children with their display keys — field names (empty for
    /// positional tuple entries) and map keys; sequence items carry no
    /// key. The read-only counterpart of [`Val::child_at`].
    pub fn kids(&self) -> Vec<(String, &Val)> {
        match self {
            Val::Atom(..) => Vec::new(),
            Val::Seq { items, .. } => items.iter().map(|it| (String::new(), &it.val)).collect(),
            Val::Struct { fields, .. } => {
                fields.iter().map(|(k, it)| (k.clone(), &it.val)).collect()
            }
            Val::Map { entries, .. } => {
                entries.iter().map(|(k, it)| (k.clone(), &it.val)).collect()
            }
        }
    }

    /// The nth child value, read-only.
    pub fn child(&self, n: usize) -> Option<&Val> {
        match self {
            Val::Atom(..) => None,
            Val::Seq { items, .. } => items.get(n).map(|it| &it.val),
            Val::Struct { fields, .. } => fields.get(n).map(|(_, it)| &it.val),
            Val::Map { entries, .. } => entries.get(n).map(|(_, it)| &it.val),
        }
    }

    /// The nth child value, whatever the container kind.
    pub fn child_at(&mut self, n: usize) -> Option<&mut Val> {
        match self {
            Val::Atom(..) => None,
            Val::Seq { items, .. } => items.get_mut(n).map(|it| &mut it.val),
            Val::Struct { fields, .. } => fields.get_mut(n).map(|(_, it)| &mut it.val),
            Val::Map { entries, .. } => entries.get_mut(n).map(|(_, it)| &mut it.val),
        }
    }

    /// Open a container or close it; atoms ignore the request.
    pub fn toggle(&mut self) {
        match self {
            Val::Seq { open, .. } | Val::Struct { open, .. } | Val::Map { open, .. } => {
                *open = !*open
            }
            Val::Atom(..) => {}
        }
    }

    /// Open every container down to depth `max`, fold what is deeper —
    /// the view's initial spread.
    pub fn open_to(&mut self, depth: usize, max: usize) {
        let children: Vec<&mut Val> = match self {
            Val::Atom(..) => return,
            Val::Seq { open, items, .. } => {
                *open = depth <= max;
                items.iter_mut().map(|it| &mut it.val).collect()
            }
            Val::Struct { open, fields, .. } => {
                *open = depth <= max;
                fields.iter_mut().map(|(_, it)| &mut it.val).collect()
            }
            Val::Map { open, entries, .. } => {
                *open = depth <= max;
                entries.iter_mut().map(|(_, it)| &mut it.val).collect()
            }
        };
        for c in children {
            c.open_to(depth + 1, max);
        }
    }
}

/// Follow a fold-path of child indices from the root, read-only.
pub fn find<'v>(root: &'v Val, path: &[u16]) -> Option<&'v Val> {
    let mut v = root;
    for &i in path {
        v = v.child(i as usize)?;
    }
    Some(v)
}

/// Follow a fold-path of child indices from the root.
pub fn walk<'v>(root: &'v mut Val, path: &[u16]) -> Option<&'v mut Val> {
    let mut v = root;
    for &i in path {
        v = v.child_at(i as usize)?;
    }
    Some(v)
}

// ---------------------------------------------------------------------------
// The parser: a pragmatic RON subset

/// A recursive-descent reader over the source bytes.
struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

/// The longest one-line text a folded container may show before the
/// bracketed ellipsis takes over.
const INLINE_MAX: usize = 44;

/// How deep the parser nests before it cries stack overflow.
const MAX_DEPTH: usize = 96;

impl<'a> Parser<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            s: src.as_bytes(),
            i: 0,
        }
    }

    /// The 1-based line of the cursor, for error messages.
    fn line(&self) -> usize {
        1 + self.s[..self.i].iter().filter(|&&b| b == b'\n').count()
    }

    fn err<T>(&self, m: &str) -> Result<T, String> {
        Err(format!("line {}: {m}", self.line()))
    }

    /// Skip whitespace, `//` lines and nestable `/* */` blocks.
    fn ws(&mut self) -> Result<(), String> {
        loop {
            while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
            if self.s[self.i..].starts_with(b"//") {
                while self.i < self.s.len() && self.s[self.i] != b'\n' {
                    self.i += 1;
                }
            } else if self.s[self.i..].starts_with(b"/*") {
                let mut depth = 0;
                while self.i < self.s.len() {
                    if self.s[self.i..].starts_with(b"/*") {
                        depth += 1;
                        self.i += 2;
                    } else if self.s[self.i..].starts_with(b"*/") {
                        depth -= 1;
                        self.i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        self.i += 1;
                    }
                }
                if depth > 0 {
                    return self.err("unterminated block comment");
                }
            } else {
                return Ok(());
            }
        }
    }

    /// Consume one comment — a `//` line or a nestable `/* */` block —
    /// verbatim, and hand back its text.
    fn comment(&mut self) -> Result<String, String> {
        let start = self.i;
        if self.s[self.i..].starts_with(b"//") {
            while self.i < self.s.len() && self.s[self.i] != b'\n' {
                self.i += 1;
            }
        } else {
            let mut depth = 0;
            while self.i < self.s.len() {
                if self.s[self.i..].starts_with(b"/*") {
                    depth += 1;
                    self.i += 2;
                } else if self.s[self.i..].starts_with(b"*/") {
                    depth -= 1;
                    self.i += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    self.i += 1;
                }
            }
            if depth > 0 {
                return self.err("unterminated block comment");
            }
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }

    /// Whitespace and comments up to the next value: the comments, in
    /// order and verbatim, joined by newlines — an entry's lead.
    fn ws_lead(&mut self) -> Result<String, String> {
        let mut cs: Vec<String> = Vec::new();
        loop {
            while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
                self.i += 1;
            }
            if self.s[self.i..].starts_with(b"//") || self.s[self.i..].starts_with(b"/*") {
                cs.push(self.comment()?);
            } else {
                return Ok(cs.join("\n"));
            }
        }
    }

    /// A comment on the current line only: spaces (never a newline)
    /// and one comment — an entry's trail. `Ok("")` when none starts
    /// before the line ends.
    fn trail(&mut self) -> Result<String, String> {
        let save = self.i;
        let mut j = self.i;
        while j < self.s.len() && (self.s[j] == b' ' || self.s[j] == b'\t' || self.s[j] == b'\r') {
            j += 1;
        }
        if self.s[j..].starts_with(b"//") || self.s[j..].starts_with(b"/*") {
            self.i = j;
            self.comment()
        } else {
            self.i = save;
            Ok(String::new())
        }
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.i).copied()
    }

    fn eat(&mut self, c: u8) -> bool {
        if self.peek() == Some(c) {
            self.i += 1;
            true
        } else {
            false
        }
    }

    /// A `path::like::Ident`, or `None` (cursor unmoved) if one starts
    /// here.
    fn ident(&mut self) -> Option<String> {
        let start = self.i;
        let first = self.peek()?;
        if !(first.is_ascii_alphabetic() || first == b'_') {
            return None;
        }
        self.i += 1;
        loop {
            while let Some(b) = self.peek() {
                if b.is_ascii_alphanumeric() || b == b'_' {
                    self.i += 1;
                } else {
                    break;
                }
            }
            // A `::` continues the path only when an identifier follows.
            if self.s[self.i..].starts_with(b"::")
                && self
                    .s
                    .get(self.i + 2)
                    .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_')
            {
                self.i += 2;
            } else {
                break;
            }
        }
        Some(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }

    /// A number: sign, digits and `_`, a fraction, an exponent, and a
    /// type suffix — kept verbatim.
    fn number(&mut self) -> Option<String> {
        let start = self.i;
        self.eat(b'-');
        self.eat(b'+');
        let mut seen = false;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() || b == b'_' || b == b'.' {
                self.i += 1;
                seen = true;
            } else if (b == b'e' || b == b'E') && seen {
                self.i += 1;
                self.eat(b'-');
                self.eat(b'+');
            } else if b.is_ascii_alphabetic() && seen {
                // A type suffix (f32, u64, ...): swallow the run.
                self.i += 1;
            } else {
                break;
            }
        }
        if seen {
            Some(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
        } else {
            self.i = start;
            None
        }
    }

    /// A `"string"` or `'c'`, captured verbatim — quotes included,
    /// escapes untouched.
    fn quoted(&mut self, q: u8) -> Result<String, String> {
        let start = self.i;
        self.i += 1; // the opening quote
        loop {
            let Some(b) = self.peek() else {
                return self.err("unterminated string");
            };
            self.i += 1;
            match b {
                b'\\' => {
                    self.i += 1;
                }
                _ if b == q => break,
                b'\n' => return self.err("unterminated string"),
                _ => {}
            }
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }

    /// One value at nesting `depth`.
    fn value(&mut self, depth: usize) -> Result<Val, String> {
        if depth > MAX_DEPTH {
            return self.err("nested too deeply");
        }
        self.ws()?;
        // RON's `!` type marker: `!Type(value)` and `!(...)` show the
        // value; the marker is the viewer's blind spot by design.
        if self.eat(b'!') {
            self.ident();
            return self.value(depth);
        }
        match self.peek() {
            Some(b'[') => {
                self.i += 1;
                let mut items = Vec::new();
                let tail: String;
                loop {
                    let lead = self.ws_lead()?;
                    if self.eat(b']') {
                        tail = lead;
                        break;
                    }
                    let val = self.value(depth + 1)?;
                    let mut trail = self.ws_lead()?;
                    let post = self.trail()?;
                    if !post.is_empty() {
                        if !trail.is_empty() {
                            trail.push(' ');
                        }
                        trail.push_str(&post);
                    }
                    items.push(Item { lead, trail, val });
                    if !self.eat(b',') {
                        tail = self.ws_lead()?;
                        if self.eat(b']') {
                            break;
                        }
                        return self.err("expected ',' or ']'");
                    }
                }
                Ok(Val::Seq {
                    open: false,
                    items,
                    tail,
                })
            }
            Some(b'(') => {
                self.i += 1;
                let (fields, tail) = self.fields(b')', depth)?;
                Ok(Val::Struct {
                    open: false,
                    head: String::new(),
                    curly: false,
                    fields,
                    tail,
                })
            }
            Some(b'{') => {
                self.i += 1;
                let mut entries = Vec::new();
                let tail: String;
                loop {
                    let lead = self.ws_lead()?;
                    if self.eat(b'}') {
                        tail = lead;
                        break;
                    }
                    let key = self.value(depth + 1)?;
                    self.ws()?;
                    if !self.eat(b':') {
                        return self.err("expected ':' after a map key");
                    }
                    let val = self.value(depth + 1)?;
                    let mut trail = self.ws_lead()?;
                    let post = self.trail()?;
                    if !post.is_empty() {
                        if !trail.is_empty() {
                            trail.push(' ');
                        }
                        trail.push_str(&post);
                    }
                    entries.push((key.display(), Item { lead, trail, val }));
                    if !self.eat(b',') {
                        tail = self.ws_lead()?;
                        if self.eat(b'}') {
                            break;
                        }
                        return self.err("expected ',' or '}'");
                    }
                }
                Ok(Val::Map {
                    open: false,
                    entries,
                    tail,
                })
            }
            Some(b'"') => {
                let s = self.quoted(b'"')?;
                Ok(Val::Atom(s, Kind::Str))
            }
            Some(b'\'') => {
                let s = self.quoted(b'\'')?;
                Ok(Val::Atom(s, Kind::Char))
            }
            Some(b) if b == b'-' || b == b'+' || b.is_ascii_digit() => {
                let Some(n) = self.number() else {
                    return self.err("bad number");
                };
                Ok(Val::Atom(n, Kind::Num))
            }
            _ => {
                let Some(p) = self.ident() else {
                    return self.err("expected a value");
                };
                self.ws()?;
                if p == "true" || p == "false" {
                    return Ok(Val::Atom(p, Kind::Bool));
                }
                if self.eat(b'(') {
                    let (fields, tail) = self.fields(b')', depth)?;
                    return Ok(Val::Struct {
                        open: false,
                        head: p,
                        curly: false,
                        fields,
                        tail,
                    });
                }
                if self.eat(b'{') {
                    let (fields, tail) = self.fields(b'}', depth)?;
                    return Ok(Val::Struct {
                        open: false,
                        head: p,
                        curly: true,
                        fields,
                        tail,
                    });
                }
                Ok(Val::Atom(p, Kind::Path))
            }
        }
    }

    /// The comma-run of entries until `close`: named (`k: v`) or
    /// positional (`v`), trailing comma allowed; with each entry its
    /// lead and trail comments, and the run's tail comments before the
    /// closing bracket.
    fn fields(&mut self, close: u8, depth: usize) -> Result<(Vec<(String, Item)>, String), String> {
        let mut fields = Vec::new();
        let tail: String;
        loop {
            let lead = self.ws_lead()?;
            if self.eat(close) {
                tail = lead;
                break;
            }
            // Named entry? An ident followed by a lone `:` — `::` is a
            // path, not a colon.
            let save = self.i;
            let named = match self.ident() {
                Some(p) => {
                    self.ws()?;
                    if self.peek() == Some(b':') && self.s.get(self.i + 1) != Some(&b':') {
                        self.i += 1;
                        Some(p)
                    } else {
                        self.i = save;
                        None
                    }
                }
                None => {
                    self.i = save;
                    None
                }
            };
            let val = self.value(depth + 1)?;
            let mut trail = self.ws_lead()?;
            let post = self.trail()?;
            if !post.is_empty() {
                if !trail.is_empty() {
                    trail.push(' ');
                }
                trail.push_str(&post);
            }
            fields.push((named.unwrap_or_default(), Item { lead, trail, val }));
            if !self.eat(b',') {
                tail = self.ws_lead()?;
                if self.eat(close) {
                    break;
                }
                return self.err("expected ',' or a closing bracket");
            }
        }
        Ok((fields, tail))
    }
}

/// A whole document: the tree plus the comments outside the root —
/// the header above it and the trailer that followed it.
#[derive(Debug, PartialEq)]
pub struct Doc {
    /// The comment run above the root, verbatim, one per line.
    pub header: String,
    /// The document's value.
    pub root: Val,
    /// The comments after the root: its same-line trail, then the
    /// whole-line run that closed the file.
    pub trailer: String,
}

/// Parse a RON document into the view's tree, keeping every comment.
pub fn parse_doc(src: &str) -> Result<Doc, String> {
    let mut p = Parser::new(src);
    let header = p.ws_lead()?;
    let root = p.value(0)?;
    let post = p.trail()?;
    let rest = p.ws_lead()?;
    let trailer = if post.is_empty() {
        rest
    } else if rest.is_empty() {
        post
    } else {
        format!("{post}\n{rest}")
    };
    if p.peek().is_some() {
        return p.err("unexpected text after the document");
    }
    Ok(Doc {
        header,
        root,
        trailer,
    })
}

/// Parse a RON document into the view's tree, comments and all.
pub fn parse(src: &str) -> Result<Val, String> {
    parse_doc(src).map(|d| d.root)
}

// ---------------------------------------------------------------------------
// The writer: the parser's counterpart, for anything that saves

/// A `"quoted"` text atom — quotes added, the house dialect spells
/// strings. Together with [`flag`], [`number`], [`record`] and
/// [`record_named`] these are the constructors for a `Val` an app
/// builds to publish: apps tell their being in values, and values
/// start somewhere.
pub fn text(s: &str) -> Val {
    Val::Atom(format!("\"{s}\""), Kind::Str)
}

/// A `true` / `false` atom.
pub fn flag(on: bool) -> Val {
    Val::Atom(on.to_string(), Kind::Bool)
}

/// A number atom, Rust's shortest round-tripping spelling.
pub fn number(v: f64) -> Val {
    Val::Atom(v.to_string(), Kind::Num)
}

/// A positional record `(a, b, ..)` — the house's tuple shape.
pub fn record(items: Vec<Val>) -> Val {
    Val::Struct {
        open: true,
        head: String::new(),
        curly: false,
        fields: items
            .into_iter()
            .map(|v| (String::new(), Item::plain(v)))
            .collect(),
        tail: String::new(),
    }
}

/// A named-fields record `(a: 1, b: 2, ..)`.
pub fn record_named(fields: &[(&str, Val)]) -> Val {
    Val::Struct {
        open: true,
        head: String::new(),
        curly: false,
        fields: fields
            .iter()
            .map(|(k, v)| ((*k).to_string(), Item::plain(v.clone())))
            .collect(),
        tail: String::new(),
    }
}

/// The value under a spoken name: the named field of a record or
/// the entry of a map. Readers who care about `ships` — not about
/// the fourth child — ask by name; apps that rename a field know
/// the reader hears it, which is what makes the telling a dialect
/// and not a coincidence of order.
pub fn field<'a>(v: &'a Val, name: &str) -> Option<&'a Val> {
    match v {
        Val::Struct { fields, .. } => fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, item)| &item.val),
        Val::Map { entries, .. } => entries
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, item)| &item.val),
        _ => None,
    }
}

/// Full RON text for a document — the counterpart to [`parse_doc`].
/// Comments survive: leads go above their entry, trails follow it on
/// the same line, tails sit before the closing bracket, and the header
/// and trailer bookend the root. Layout is otherwise canonical: one
/// element per line whenever a container has children, trailing commas
/// everywhere; the original line breaks and the fold state do not
/// survive. Re-parsing the text yields the same document.
pub fn to_text_doc(header: &str, root: &Val, trailer: &str) -> String {
    let mut out = String::new();
    for line in comment_lines(header) {
        out.push_str(line);
        out.push('\n');
    }
    write_val(root, 0, &mut out);
    let one_line = !trailer.contains('\n');
    if one_line {
        if !trailer.trim().is_empty() {
            out.push(' ');
            out.push_str(trailer.trim());
        }
        out.push('\n');
    } else {
        out.push('\n');
        for line in comment_lines(trailer) {
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

/// Full RON text for a value, comments and all (no header or trailer).
pub fn to_text(root: &Val) -> String {
    to_text_doc("", root, "")
}

/// The non-blank lines of a comment run.
fn comment_lines(run: &str) -> impl Iterator<Item = &str> {
    run.lines().map(str::trim_end).filter(|l| !l.is_empty())
}

/// A child's written form: its lead's lines, its own line, its trail.
type WrittenKid = (String, String, String);

/// One value at `depth`, into `out`.
fn write_val(v: &Val, depth: usize, out: &mut String) {
    let (head, o, c, kids, tail): (String, char, char, Vec<WrittenKid>, String) = match v {
        Val::Atom(s, _) => {
            out.push_str(s);
            return;
        }
        Val::Seq { items, tail, .. } => (
            String::new(),
            '[',
            ']',
            items
                .iter()
                .map(|it| {
                    let mut t = String::new();
                    write_val(&it.val, depth + 1, &mut t);
                    (it.lead.clone(), t, it.trail.clone())
                })
                .collect(),
            tail.clone(),
        ),
        Val::Struct {
            head,
            curly,
            fields,
            tail,
            ..
        } => (
            head.clone(),
            if *curly { '{' } else { '(' },
            if *curly { '}' } else { ')' },
            fields
                .iter()
                .map(|(k, it)| {
                    let mut t = String::new();
                    if !k.is_empty() {
                        t.push_str(k);
                        t.push_str(": ");
                    }
                    write_val(&it.val, depth + 1, &mut t);
                    (it.lead.clone(), t, it.trail.clone())
                })
                .collect(),
            tail.clone(),
        ),
        Val::Map { entries, tail, .. } => (
            String::new(),
            '{',
            '}',
            entries
                .iter()
                .map(|(k, it)| {
                    let mut t = String::new();
                    t.push_str(k);
                    t.push_str(": ");
                    write_val(&it.val, depth + 1, &mut t);
                    (it.lead.clone(), t, it.trail.clone())
                })
                .collect(),
            tail.clone(),
        ),
    };
    out.push_str(&head);
    if kids.is_empty() && tail.trim().is_empty() {
        out.push(o);
        out.push(c);
        return;
    }
    out.push(o);
    out.push('\n');
    let pad = "    ".repeat(depth + 1);
    for (lead, text, trail) in &kids {
        for line in comment_lines(lead) {
            out.push_str(&pad);
            out.push_str(line);
            out.push('\n');
        }
        out.push_str(&pad);
        out.push_str(text);
        out.push(',');
        if !trail.trim().is_empty() {
            out.push(' ');
            out.push_str(trail.trim());
        }
        out.push('\n');
    }
    for line in comment_lines(&tail) {
        out.push_str(&pad);
        out.push_str(line);
        out.push('\n');
    }
    out.push_str(&"    ".repeat(depth));
    out.push(c);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped sample, embedded so the parser is tested against its
    /// real target.
    const GARDEN: &str = include_str!("../assets/ron/garden.ron");

    #[test]
    fn the_garden_parses_with_every_flavour() {
        let root = parse(GARDEN).expect("the sample must parse");
        let Val::Struct { head, fields, .. } = &root else {
            panic!("the root is an anonymous struct");
        };
        assert!(head.is_empty());
        let names: Vec<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(names, ["gardener", "bench", "watering", "chores"]);
    }

    #[test]
    fn enums_arrive_as_paths_heads_and_variants() {
        let root = parse(GARDEN).unwrap();
        // A unit enum is a path atom; a newtype enum is a one-field
        // struct named by its path.
        let Val::Struct { fields, .. } = &root else {
            bug()
        };
        let chores = &fields[3].1.val;
        let Val::Map { entries, .. } = chores else {
            bug()
        };
        assert_eq!(entries[0].0, "\"mon\"");
        let Val::Struct { head, fields, .. } = &entries[0].1.val else {
            bug()
        };
        assert_eq!(head, "Chores");
        let weed = &fields[1].1.val;
        let Val::Struct { head, fields, .. } = weed else {
            bug()
        };
        assert_eq!(head, "Kind::Bug");
        assert_eq!(fields.len(), 1);
        let Val::Struct { head, .. } = &fields[0].1.val else {
            bug()
        };
        assert_eq!(head, "Count");
        // And a struct variant with curly fields.
        let Val::Struct { fields, .. } = &root else {
            bug()
        };
        let Val::Struct {
            head,
            curly,
            fields,
            ..
        } = &fields[2].1.val
        else {
            bug()
        };
        assert_eq!(head, "Watering::PingPong");
        assert!(curly);
        assert_eq!(fields.len(), 2);
    }

    #[test]
    fn atoms_keep_their_text_and_kind() {
        let v = parse(
            "!(name: \"Per \\\"Q\\\"\", n: -1_000, f: 6.5e-2, t: true, u: Kind::Bee, c: 'x')",
        )
        .unwrap();
        let Val::Struct { fields, .. } = &v else {
            bug()
        };
        let atom = |i: usize| match &fields[i].1.val {
            Val::Atom(s, k) => (s.clone(), *k),
            _ => bug(),
        };
        assert_eq!(atom(0), ("\"Per \\\"Q\\\"\"".into(), Kind::Str));
        assert_eq!(atom(1), ("-1_000".into(), Kind::Num));
        assert_eq!(atom(2), ("6.5e-2".into(), Kind::Num));
        assert_eq!(atom(3), ("true".into(), Kind::Bool));
        assert_eq!(atom(4), ("Kind::Bee".into(), Kind::Path));
        assert_eq!(atom(5), ("'x'".into(), Kind::Char));
    }

    #[test]
    fn comments_and_trailing_commas_are_noise() {
        let v = parse("/* a /* nested */ block */ ( a: 1, /* inline */ b: [1, 2,], ) // trailing")
            .unwrap();
        let Val::Struct { fields, .. } = &v else {
            bug()
        };
        assert_eq!(fields.len(), 2);
        let Val::Seq { items, .. } = &fields[1].1.val else {
            bug()
        };
        assert_eq!(items.len(), 2);
    }

    #[test]
    fn errors_carry_a_line() {
        let err = parse("(a: 1,\nb: [1,,])\n").unwrap_err();
        assert!(err.starts_with("line 2:"), "{err}");
        let err = parse("(a: 1) junk").unwrap_err();
        assert!(err.starts_with("line 1:"), "{err}");
        assert!(parse("/* open").is_err());
    }

    /// Every test's panic door when the tree is not the expected shape.
    fn bug() -> ! {
        panic!("unexpected shape");
    }

    #[test]
    fn to_text_round_trips_the_garden() {
        let one = parse_doc(GARDEN).expect("the garden parses");
        let text = to_text_doc(&one.header, &one.root, &one.trailer);
        let two = parse_doc(&text).unwrap_or_else(|err| panic!("re-parse failed: {err}\n{text}"));
        assert_eq!(one, two);
    }

    #[test]
    fn comments_survive_the_round_trip() {
        let src = "// header\n(\n    // lead a\n    a: [1, 2], // trail a\n    b: (3.0, 4.0),\n    // tail note\n)\n// trailer\n";
        let d = parse_doc(src).expect("parses");
        assert_eq!(d.header, "// header");
        assert_eq!(d.trailer, "// trailer");
        let text = to_text_doc(&d.header, &d.root, &d.trailer);
        for wanted in [
            "// header",
            "// lead a",
            "// trail a",
            "// tail note",
            "// trailer",
        ] {
            assert!(text.contains(wanted), "lost {wanted}:\n{text}");
        }
        let again = parse_doc(&text).unwrap_or_else(|err| panic!("re-parse: {err}\n{text}"));
        assert_eq!(d, again);
    }
}
