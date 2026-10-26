//! The telling: how a running application announces itself to its
//! guests.
//!
//! In frost the truth is a running program — the scene tree is a
//! frame artifact, manufactured and discarded — so a tool cannot
//! *extract* an application's state of affairs; it can only be
//! *told*. This module is the apparatus of telling: an announce
//! card in a well-known folder (tools enumerate it the way a
//! debugger lists processes), and a live file the application
//! re-publishes when its being changes.
//!
//! The telling is transition-driven and carries structure only.
//! The clock is a value-key dirty check — the serialization *is*
//! the snapshot, so comparing against the last published body is
//! nearly free and a forgotten flag cannot hide a change. Every
//! write carries a monotonic `rev`, and the first write happens at
//! boot: coming into existence is a transition. Idle apps write
//! nothing, ever — this is an autosave with manners, not a stream.
//!
//! Liveness stays the card's pid question — change and life are
//! different questions and do not mix in one file. The card dies
//! with the app (its `Drop`) and stale cards — crashed apps — are
//! pruned by pid liveness when a new app opens its folder. The
//! dialect is the house RON: tools read the tree with the parser
//! they already have, and tools write documents only.

use std::fs;
use std::path::{Path, PathBuf};

/// The apparatus one application uses to tell: the card it leaves
/// in the folder, the live file it publishes, and the last body
/// published — the value key the dirty check turns on.
#[derive(Debug)]
pub struct Teller {
    name: String,
    pid: u32,
    card: PathBuf,
    live: PathBuf,
    last: Option<String>,
    rev: u64,
}

impl Teller {
    /// Open the telling in `dir`: create the folder, prune the cards
    /// of the departed, and pin this app's card. The card names the
    /// live file's location; the live file itself waits for the
    /// first publish — which should come at boot, because coming
    /// into existence is a transition.
    pub fn open(dir: impl Into<PathBuf>, name: &str) -> Self {
        let dir = dir.into();
        let _ = fs::create_dir_all(&dir);
        let pid = std::process::id();
        prune(&dir, pid);
        let card = dir.join(format!("{name}-{pid}.card.ron"));
        let live = dir.join(format!("{name}-{pid}.live.ron"));
        let root = crate::ron::record_named(&[
            ("name", crate::ron::text(name)),
            ("pid", crate::ron::number(f64::from(pid))),
            (
                "asset_root",
                crate::ron::text(
                    &std::env::current_dir()
                        .map(|d| d.display().to_string())
                        .unwrap_or_default(),
                ),
            ),
            ("live_file", crate::ron::text(&live.display().to_string())),
        ]);
        let header = format!(
            "// {name} is running — this card dies with the app, and is pruned while it sleeps\n"
        );
        let _ = fs::write(&card, crate::ron::to_text_doc(&header, &root, ""));
        Self {
            name: name.into(),
            pid,
            card,
            live,
            last: None,
            rev: 0,
        }
    }

    /// The live file's path — the place guests read the being.
    pub fn live_file(&self) -> &Path {
        &self.live
    }

    /// Tell the app's being: serialize, compare against the last
    /// published body, and write only on change — every write
    /// bumps `rev`. The body excludes the `rev` line, so a
    /// heartbeat-shaped write is impossible: identical means silent.
    /// Returns whether the file changed.
    pub fn publish(&mut self, being: &crate::ron::Val) -> bool {
        let body = crate::ron::to_text_doc("", being, "");
        if self.last.as_deref() == Some(body.as_str()) {
            return false;
        }
        self.rev += 1;
        let header = format!(
            "// {} tells its being — rev {}, pid {}\n",
            self.name, self.rev, self.pid
        );
        match fs::write(&self.live, crate::ron::to_text_doc(&header, being, "")) {
            Ok(()) => {
                self.last = Some(body);
                log::debug!("telling: '{}' published rev {}", self.name, self.rev);
                true
            }
            Err(err) => {
                // The key stays unset so the next frame retries;
                // a dev folder that cannot be written is worth one
                // warning, then silence until it heals.
                log::warn!("telling: '{}' cannot speak ({err})", self.name);
                false
            }
        }
    }
}

/// A card's answer: what a running application announced itself
/// to be, read back from its card. The guest's view of `ls` on the
/// telling folder — a debugger's process list, wearing the house
/// dialect.
#[derive(Debug, Clone, PartialEq)]
pub struct Announcement {
    /// The application's own name for itself.
    pub name: String,
    /// The author's process — liveness is still this question alone.
    pub pid: u32,
    /// Where the app was working: the root its relative paths speak
    /// from.
    pub asset_root: std::path::PathBuf,
    /// The file the app's tellings land in.
    pub live_file: PathBuf,
    /// The card itself — its absence is how the app's death is
    /// noticed.
    pub card: PathBuf,
}

/// The latest telling, read: the revision from the file's header
/// and the being from its root. What a guest knows; how the app
/// came to tell it stays the app's business.
#[derive(Debug)]
pub struct Tiding {
    /// The monotonic count of tellings published.
    pub rev: u64,
    /// The structure the app told, whole.
    pub being: crate::ron::Val,
}

impl Teller {
    /// List the folder as a debugger lists processes: the surviving
    /// cards of living apps, by name then pid. Prunes the departed
    /// first, because listing and healing are the same question
    /// here — who is gone.
    pub fn announce(dir: impl AsRef<Path>) -> Vec<Announcement> {
        let dir = dir.as_ref();
        prune(dir, std::process::id());
        let mut out = Vec::new();
        let Ok(entries) = fs::read_dir(dir) else {
            return out;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let is_card = path
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".card.ron"));
            if !is_card {
                continue;
            }
            let Ok(src) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(doc) = crate::ron::parse_doc(&src) else {
                continue;
            };
            let name = crate::ron::field(&doc.root, "name").map(spoken);
            let pid =
                crate::ron::field(&doc.root, "pid").and_then(|v| v.inline().parse::<u32>().ok());
            let asset_root = crate::ron::field(&doc.root, "asset_root").map(spoken);
            let live_file = crate::ron::field(&doc.root, "live_file").map(spoken);
            let (Some(name), Some(pid), Some(asset_root), Some(live_file)) =
                (name, pid, asset_root, live_file)
            else {
                continue;
            };
            if alive(pid) {
                out.push(Announcement {
                    name,
                    pid,
                    asset_root: PathBuf::from(asset_root),
                    live_file: PathBuf::from(live_file),
                    card: path,
                });
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name).then(a.pid.cmp(&b.pid)));
        out
    }

    /// Read an announcement's latest telling: `None` when the file
    /// refuses to parse or cannot be found — from either side, a
    /// relative path is resolved against the card's own answers,
    /// so guest and app need not share a working directory.
    pub fn tiding(what: &Announcement) -> Option<Tiding> {
        let mut candidates = vec![what.live_file.clone()];
        if what.live_file.is_relative() {
            candidates.push(what.asset_root.join(&what.live_file));
            candidates.push(what.card.parent()?.join(&what.live_file));
        }
        let src = candidates
            .iter()
            .find_map(|path| fs::read_to_string(path).ok())?;
        let rev = src.lines().find_map(|line| {
            let rest = line.split_once("rev ")?.1;
            let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse().ok()
        })?;
        let doc = crate::ron::parse_doc(&src).ok()?;
        Some(Tiding {
            rev,
            being: doc.root,
        })
    }
}

/// A quoted atom's spoken text: the value's one-line inline, minus
/// the quotes the dialect wears.
fn spoken(v: &crate::ron::Val) -> String {
    v.inline().trim_matches('"').to_string()
}

impl Drop for Teller {
    fn drop(&mut self) {
        // The card is the live half of the announcement: gone with
        // the app, so a folder full of cards is a list of what is
        // running — as accurate as exits are honored.
        let _ = fs::remove_file(&self.card);
    }
}

/// Remove whatever an author no longer with us left: cards whose
/// pid is gone, and the live files of departed pids too — orphaned
/// documents (their card died with the app, the document stayed)
/// are folder litter as much as crashed apps' cards are. A process
/// we cannot signal looks departed — the folder is dev furniture,
/// and the cost of that lie is one deleted file a live stranger
/// republishes on its next telling.
fn prune(dir: &Path, me: u32) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(name) => name.to_string(),
            None => continue,
        };
        let stem = match name
            .strip_suffix(".card.ron")
            .or_else(|| name.strip_suffix(".live.ron"))
        {
            Some(stem) => stem.to_string(),
            None => continue,
        };
        let Some((_, pid)) = stem.rsplit_once('-') else {
            continue;
        };
        let Ok(pid) = pid.parse::<u32>() else {
            continue;
        };
        if pid != me && !alive(pid) {
            let _ = fs::remove_file(dir.join(format!("{stem}.card.ron")));
            let _ = fs::remove_file(dir.join(format!("{stem}.live.ron")));
            log::debug!("telling: pruned what the departed {stem} left behind");
        }
    }
}

#[cfg(unix)]
fn alive(pid: u32) -> bool {
    // Signal zero, asked of the utility instead of the raw call:
    // `/bin/kill -0` checks the pid and delivers nothing, answering
    // the whole question through its exit status — no syscall, no
    // unsafe, and none of `kill`'s sharp edges: a pid past the
    // platform's ceiling and a pid reused by somebody else's
    // process (an EPERM the raw call also answered as gone) both
    // fail the status the same way the raw call judged them gone.
    // If the probe itself cannot run, the card gets the benefit of
    // the doubt: a broken question prunes nothing.
    std::process::Command::new("/bin/kill")
        .arg("-0")
        .arg(pid.to_string())
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_or(true, |status| status.success())
}

#[cfg(windows)]
fn alive(_pid: u32) -> bool {
    // No signal-zero without opening a process handle; on Windows a
    // card is trusted to have an author, and a crash leaves one to
    // clean by hand. Honest, narrow, dev-only.
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(who: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("frost_tell_{who}_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    /// The opening of a telling: boot publishes rev 1 to the live
    /// file, the card names the app and its pid — and the same being
    /// published twice writes once, because an idle app writes
    /// nothing, ever.
    #[test]
    fn a_booting_app_publishes_rev_one_and_silence_is_free() {
        let dir = scratch("boot");
        let mut tell = Teller::open(&dir, "boot");
        let being = crate::ron::record_named(&[("told", crate::ron::text("boot"))]);
        assert!(tell.publish(&being));
        let text = fs::read_to_string(tell.live_file()).unwrap();
        assert!(text.contains("rev 1"), "{text}");
        assert!(dir.join(format!("boot-{}.card.ron", tell.pid)).exists());
        assert!(!tell.publish(&being), "the same body is silence");
        assert_eq!(text, fs::read_to_string(tell.live_file()).unwrap());
        fs::remove_dir_all(&dir).ok();
    }

    /// Change is counted: a different being is a new rev, and the
    /// rev rides the file so a guest that coalesces watcher events
    /// still sees that something happened, and in what order.
    #[test]
    fn a_changed_being_bumps_the_rev() {
        let dir = scratch("rev");
        let mut tell = Teller::open(&dir, "rev");
        assert!(tell.publish(&crate::ron::record_named(&[(
            "beds",
            crate::ron::flag(false)
        )])));
        assert!(tell.publish(&crate::ron::record_named(&[(
            "beds",
            crate::ron::flag(true)
        )])));
        assert!(
            fs::read_to_string(tell.live_file())
                .unwrap()
                .contains("rev 2")
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// The folder heals itself: a card whose author is gone — a
    /// crashed app, here a pid that cannot be ours — is pruned
    /// together with the live file it pointed at, when any new app
    /// opens the folder.
    #[test]
    fn a_departed_card_is_pruned_with_its_file() {
        let dir = scratch("prune");
        fs::create_dir_all(&dir).unwrap();
        let ghost = "ghost-4294967290";
        fs::write(dir.join(format!("{ghost}.card.ron")), "// stale\n").unwrap();
        fs::write(dir.join(format!("{ghost}.live.ron")), "// stale\n").unwrap();
        let orphan = "orphan-4294967291";
        fs::write(
            dir.join(format!("{orphan}.live.ron")),
            "// document without a card\n",
        )
        .unwrap();
        let tell = Teller::open(&dir, "prune");
        assert!(!dir.join(format!("{ghost}.card.ron")).exists());
        assert!(!dir.join(format!("{ghost}.live.ron")).exists());
        assert!(
            !dir.join(format!("{orphan}.live.ron")).exists(),
            "orphans heal too"
        );
        assert!(dir.join(format!("prune-{}.card.ron", tell.pid)).exists());
        fs::remove_dir_all(&dir).ok();
    }

    /// The card is the live half of the announcement: it dies with
    /// the telling, and the last published being stays behind —
    /// a dead app leaves a document, not an announcement.
    #[test]
    fn the_card_dies_and_the_document_stays() {
        let dir = scratch("drop");
        {
            let mut tell = Teller::open(&dir, "drop");
            assert!(tell.publish(&crate::ron::record_named(&[(
                "told",
                crate::ron::text("drop")
            )])));
            assert!(dir.join(format!("drop-{}.card.ron", tell.pid)).exists());
        }
        assert!(
            !dir.join(format!("drop-{}.card.ron", std::process::id()))
                .exists()
        );
        assert!(
            dir.join(format!("drop-{}.live.ron", std::process::id()))
                .exists()
        );
        fs::remove_dir_all(&dir).ok();
    }

    /// The guest's side of the protocol, end to end in a scratch
    /// folder: a card of a living pid is listed (a ghost's is not),
    /// and its telling reads back with rev from the header and
    /// structure from the root.
    #[test]
    fn a_guest_reads_the_folder_and_the_telling() {
        let dir = scratch("guest");
        let me = std::process::id();
        let tell = Teller::open(&dir, "guest");
        assert_eq!(tell.name, "guest");
        fs::write(
            dir.join("ghost-4294967290.card.ron"),
            "// stale\n(\n    name: \"ghost\",\n    pid: 4294967290,\n    asset_root: \"/nowhere\",\n    live_file: \"ghost.live.ron\",\n)\n",
        )
        .unwrap();
        let cards = Teller::announce(&dir);
        assert_eq!(cards.len(), 1, "only the living are listed");
        assert_eq!(cards[0].name, "guest");
        assert_eq!(cards[0].pid, me);
        // The app had not published yet — nothing to read.
        assert!(Teller::tiding(&cards[0]).is_none());
        // Now it tells, by hand: header rev and rooted being.
        let being = crate::ron::record_named(&[("told", crate::ron::text("guest"))]);
        let text = format!(
            "// guest tells its being — rev 7, pid {me}\n{}",
            crate::ron::to_text_doc("", &being, "")
        );
        fs::write(tell.live_file(), &text).unwrap();
        let tiding = Teller::tiding(&cards[0]).expect("the telling reads");
        assert_eq!(tiding.rev, 7);
        assert_eq!(
            crate::ron::field(&tiding.being, "told")
                .map(|v| v.inline())
                .as_deref(),
            Some("\"guest\"")
        );
        fs::remove_dir_all(&dir).ok();
    }
}
