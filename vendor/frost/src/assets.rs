//! Assets by path. Game code never says `include_bytes!`: it says
//! *give me `assets/sprites/grass.png`* — the same words a tileset or
//! map document speaks — and this module answers, from the table
//! compiled into the binary or from a loose file under an admitted
//! root.
//!
//! The design is one request and two worlds. Release ships one
//! binary: the [`Table`] answers every path, its keys written in the
//! same root-relative dialect the documents use, and no asset file
//! needs to exist. Development runs the same code with a root
//! admitted: the request is joined to that root and read, so an edit
//! saved in `sprite_util` lands in the next load — and a folder left
//! beside a shipped binary is the update channel the single binary
//! was promised. Nothing resolves a `CARGO_MANIFEST_DIR` at runtime:
//! the root is an input the caller hands in, compile-time by
//! convention and dev-only by practice. No feature flag forks the
//! two; the debug binary carries its table and prefers the loose
//! file, which is the price of one code path, paid gladly.
//!
//! Paths are the identity now, and two laws keep that honest. They
//! are always root-relative and always `/`-joined — the key and the
//! embed and the document reference are one string. And matching is
//! case-exact all the way down, even on the filesystems that would
//! forgive a case slip: a wrong-case path must refuse in development
//! exactly when it refuses in release, or the two worlds stop being
//! the same world at their worst moment.

use std::path::{Path, PathBuf};

/// The table an app ships: every asset by path, every path by bytes.
/// The key is root-relative, `/`-joined, and case-exact — the same
/// words the app's documents speak and the same words
/// `include_bytes!` names the file with at build time, so key,
/// embed, and request are one dialect from the desk to the binary.
pub type Table = &'static [(&'static str, &'static [u8])];

/// The resolver behind the one request: the admitted roots first,
/// the compiled table beneath them, `None` beneath both.
#[derive(Clone, Default)]
pub struct Assets {
    /// The bytes compiled into this binary, keyed by path.
    table: Table,
    /// The admitted roots, first winning: a root that does not exist
    /// — the normal case beside a shipped binary — answers nothing
    /// and asks nothing of the walk.
    roots: Vec<PathBuf>,
}

/// The request as the dialect means it: `\` read as `/`, a leading
/// `./` stripped, nothing else touched — a key is a key.
fn canonical(path: &str) -> String {
    path.replace('\\', "/").trim_start_matches("./").to_string()
}

/// The file a path names under `root` — if every component of it
/// exists with the exact case the request spoke. `read_dir` is the
/// only judge: on the forgiving filesystems a plain `exists()` would
/// nod at `Grass.png` for `grass.png`, and a release table would not.
fn exact_under(root: &Path, path: &str) -> Option<PathBuf> {
    let mut dir = root.to_path_buf();
    for comp in path.split('/').filter(|c| !c.is_empty() && *c != ".") {
        let entries = std::fs::read_dir(&dir).ok()?;
        let found = entries.flatten().find(|e| e.file_name() == *comp)?;
        dir = found.path();
    }
    Some(dir)
}

impl Assets {
    /// The resolver of a shipped world: the table, and nothing loose.
    #[must_use]
    pub fn embedded(table: Table) -> Self {
        Self {
            table,
            roots: Vec::new(),
        }
    }

    /// Admit one asset root: a request for `assets/sprites/grass.png`
    /// reads `<root>/assets/sprites/grass.png` when that exact path
    /// stands, and a loose file outranks the table's entry of the
    /// same path. Called in development; harmless anywhere else,
    /// where the root does not answer and the table stands alone.
    #[must_use]
    pub fn with_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        if !dir.is_dir() {
            log::debug!("assets: root '{}' speaks nothing", dir.display());
        }
        self.roots.push(dir);
        self
    }

    /// The bytes a path means now, whatever format they arrive in:
    /// the loose file if one stands at it, the table otherwise,
    /// nothing if neither does. A loose file that cannot be read —
    /// mid-save, half-written — falls to the table rather than going
    /// silent.
    #[must_use]
    pub fn bytes(&self, path: &str) -> Option<std::borrow::Cow<'static, [u8]>> {
        self.asset(path, &[])
    }

    /// The bytes a path means, if it wears one of the accepted
    /// formats. The extension filter kept its post from the stem
    /// days and earns its keep in its new one: an app that asks for
    /// a sprite by a path that names a `.ron` is holding the wrong
    /// document, and the filter lets the table answer such a mix-up
    /// rather than a decoder discover it.
    #[must_use]
    pub fn asset(&self, path: &str, exts: &[&str]) -> Option<std::borrow::Cow<'static, [u8]>> {
        let path = canonical(path);
        let speaks = |file: &str| {
            exts.is_empty()
                || Path::new(file).extension().is_some_and(|e| {
                    let e = e.to_string_lossy();
                    exts.iter().any(|x| e.eq_ignore_ascii_case(x))
                })
        };
        for root in &self.roots {
            let Some(file) = exact_under(root, &path) else {
                continue;
            };
            if !speaks(&path) {
                continue;
            }
            match std::fs::read(&file) {
                Ok(bytes) => return Some(std::borrow::Cow::Owned(bytes)),
                Err(err) => log::warn!("assets: '{}' unreadable: {err}", file.display()),
            }
        }
        self.table
            .iter()
            .find(|(key, _)| canonical(key) == path)
            .map(|(_, bytes)| std::borrow::Cow::Borrowed(*bytes))
    }

    /// The paths the table ships: an app's guard test walks its maps
    /// against these — [`crate::map::guard_documents`] asks the
    /// whole chain, map to tileset to sheet, of the table itself.
    pub fn names(&self) -> impl Iterator<Item = &'static str> {
        self.table.iter().map(|(path, _)| *path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIP: Table = &[
        ("sprites/tiles.png", b"table bytes"),
        ("sprites/grass.png", b"table grass"),
    ];

    /// The shipped world answers by path, and a path nobody ships
    /// answers nothing — a missing asset is a miss, not a guess.
    #[test]
    fn the_table_answers_by_path() {
        let assets = Assets::embedded(SHIP);
        assert_eq!(
            assets.bytes("sprites/tiles.png").as_deref(),
            Some(&b"table bytes"[..])
        );
        assert!(assets.bytes("sprites/barn.png").is_none());
        // The dialect is forgiving about spelling, never about truth:
        // backslashes and a leading ./ name the same key.
        assert_eq!(
            assets.bytes(".\\sprites\\tiles.png").as_deref(),
            Some(&b"table bytes"[..])
        );
    }

    /// The paths the table ships are askable: this is what an app's
    /// guard test walks its documents against.
    #[test]
    fn the_guard_can_ask_what_ships() {
        let paths: Vec<_> = Assets::embedded(SHIP).names().collect();
        assert_eq!(paths, vec!["sprites/tiles.png", "sprites/grass.png"]);
    }

    /// The sidecar trap, in its new dialect: a path that names a
    /// `.ron` is not a sprite, and the format filter lets the table
    /// answer where the loose document sits.
    #[test]
    fn a_sidecar_never_dresses_a_sprite() {
        let dir = std::env::temp_dir().join(format!("frost_assets_sidecar_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sprites")).unwrap();
        std::fs::write(dir.join("sprites/tiles.png.ron"), b"(atlas: [2, 8])").unwrap();
        const SIDE: Table = &[("sprites/tiles.png", b"table bytes")];
        let assets = Assets::embedded(SIDE).with_dir(&dir);
        // The sprite's path stands nowhere loose — only its sidecar
        // does — and the sidecar, exact-path and format-filtered
        // both, never dresses the sprite: the table's picture answers.
        assert_eq!(
            assets.asset("sprites/tiles.png", &["png"]).as_deref(),
            Some(&b"table bytes"[..])
        );
        // The sidecar answers its own path, in its own format only.
        assert_eq!(
            assets.asset("sprites/tiles.png.ron", &["ron"]).as_deref(),
            Some(&b"(atlas: [2, 8])"[..])
        );
        assert!(assets.asset("sprites/tiles.png.ron", &["png"]).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A loose file outranks its table entry — the dev loop's edit
    /// is what the next load sees — while paths without a loose
    /// file keep answering from the table.
    #[test]
    fn a_loose_file_outranks_the_table() {
        let dir = std::env::temp_dir().join(format!("frost_assets_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sprites")).unwrap();
        std::fs::write(dir.join("sprites/grass.png"), b"loose grass").unwrap();
        let assets = Assets::embedded(SHIP).with_dir(&dir);
        assert_eq!(
            assets.bytes("sprites/grass.png").as_deref(),
            Some(&b"loose grass"[..])
        );
        assert_eq!(
            assets.bytes("sprites/tiles.png").as_deref(),
            Some(&b"table bytes"[..])
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Case is a dialect law, not a filesystem courtesy: a request
    /// whose case slips refuses on every platform — exactly as the
    /// release table would refuse it. (On case-sensitive filesystems
    /// the loose file simply never stood; on forgiving ones the walk
    /// checks each name against its folder and declines the forgery.)
    #[test]
    fn a_case_slip_refuses_rather_than_forges() {
        let dir = std::env::temp_dir().join(format!("frost_assets_case_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("sprites")).unwrap();
        std::fs::write(dir.join("sprites/grass.png"), b"loose grass").unwrap();
        const NOTHING: Table = &[];
        let assets = Assets::embedded(NOTHING).with_dir(&dir);
        assert_eq!(
            assets.bytes("sprites/grass.png").as_deref(),
            Some(&b"loose grass"[..])
        );
        assert!(
            assets.bytes("sprites/Grass.png").is_none(),
            "the forgiving filesystem nodded; the walk did not"
        );
        assert!(
            assets.bytes("Sprites/grass.png").is_none(),
            "same law, folder components"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
