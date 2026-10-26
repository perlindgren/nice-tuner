//! Watching a file for change: the ladder's shared sense.
//!
//! Every hot tier climbs the same rung-shape — notice, re-read, swap
//! in — and noticing is this module's whole job: remember a file's
//! stamp, and answer `changed()` truthfully whenever it differs.
//! Polling metadata is a deliberate modesty: one `stat` per watched
//! file per frame costs microseconds, carries no OS event machinery,
//! and works identically for data, textures, maps, and shaders — the
//! tiers differ in what they re-read, never in how they notice.
//!
//! A file that does not exist is watched like one that does:
//! appearing is a change, and so is vanishing — a loose development
//! file removed beside a shipped binary tells the app to fall back
//! to what the binary carries. On targets without a filesystem every
//! answer is `false`: a shipped world does not change under itself.

/// One file, watched by its stamp: `changed()` reports that the file
/// on disk is not the one last seen — rewritten, appeared, or gone.
#[derive(Clone, Debug)]
pub struct Watch {
    /// The file being watched, kept for the log's sake.
    target: std::path::PathBuf,
    /// The stamp at the last look, `None` while the file did not
    /// exist: no stamp is a stamp of its own.
    stamp: Option<std::time::SystemTime>,
}

impl Watch {
    /// Watch a file from its state right now: the first `changed()`
    /// answers for changes after this moment, not before it.
    #[must_use]
    pub fn new(target: impl Into<std::path::PathBuf>) -> Self {
        let target = target.into();
        Self {
            stamp: stamp(&target),
            target,
        }
    }

    /// True once per change: the file was rewritten, appeared, or
    /// vanished since the last look. Every other frame: false.
    pub fn changed(&mut self) -> bool {
        let now = stamp(&self.target);
        let changed = now != self.stamp;
        if changed {
            self.stamp = now;
            log::debug!("watch: '{}' changed", self.target.display());
        }
        changed
    }
}

/// The file's stamp — its last-modified time — or `None` while it
/// cannot be asked: absent, or a target without a filesystem. Both
/// answer the same, and both are honest: nothing has changed that
/// can be seen.
fn stamp(target: &std::path::Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(target).and_then(|m| m.modified()).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A scratch file with a name of its own, gone when the test is.
    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("frost_watch_{name}_{}", std::process::id()))
    }

    /// The quiet case first: a file nobody touches never reports.
    #[test]
    fn an_untouched_file_is_never_a_change() {
        let path = scratch("quiet");
        std::fs::write(&path, b"one").unwrap();
        let mut w = Watch::new(&path);
        assert!(!w.changed());
        assert!(!w.changed());
        std::fs::remove_file(&path).ok();
    }

    /// A rewrite is a change, once: the watcher notices and settles.
    #[test]
    fn a_rewrite_is_one_change_not_an_argument() {
        let path = scratch("rewrite");
        std::fs::write(&path, b"one").unwrap();
        let mut w = Watch::new(&path);
        std::fs::write(&path, b"two").unwrap();
        assert!(w.changed());
        assert!(!w.changed());
        std::fs::remove_file(&path).ok();
    }

    /// Appearing is a change — the development file written after
    /// the app booted — and so is vanishing, which sends the app
    /// back to what its binary carries.
    #[test]
    fn appearing_and_vanishing_are_changes_too() {
        let path = scratch("appear");
        std::fs::remove_file(&path).ok();
        let mut w = Watch::new(&path);
        assert!(!w.changed());
        std::fs::write(&path, b"hello").unwrap();
        assert!(w.changed());
        std::fs::remove_file(&path).ok();
        assert!(w.changed());
    }
}
