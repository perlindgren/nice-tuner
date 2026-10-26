//! The store: text kept under a name until the next launch. The design
//! is `frost::Assets` turned around. The assets table answers *give me
//! the bytes this name ships with*, and cannot write; the store answers
//! *hold this text under this name for next time*, and cannot ship.
//! The same noun survives both — the name — and the answer is a file
//! beside the launch folder, the shape settings and saves have always
//! had. A game asks once, by name: `antifrost.progress.ron` is a file
//! on the desktop, and the code that reads it is one code.
//!
//! What travels is text, because what wants keeping is text — Ron
//! documents, save files, a settings page the player can read and fix
//! by hand. Binary rides no such promise: a save that outlives a GPU
//! format should not be bytes with a layout nobody remembers. Games
//! that want binary keep it as text they encode themselves.
//!
//! Failure is a returned shrug, never a popped frame: no disc to write
//! to costs persistence, not the game.

/// Reads the text kept under `name`, if anything is. A name never kept
/// and an unreadable file answer the same way: `None`, the words for
/// *nothing to remember yet*.
pub fn load(name: &str) -> Option<String> {
    let path = std::env::current_dir().unwrap_or_default().join(name);
    std::fs::read_to_string(path).ok()
}

/// Keeps `text` under `name` until the next launch — over what stood
/// there. The `String` error is a message for the log line, not a
/// condition to branch on.
pub fn save(name: &str, text: &str) -> Result<(), String> {
    let path = std::env::current_dir().unwrap_or_default().join(name);
    std::fs::write(path, text).map_err(|err| err.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The test's own name, so a run that died before its cleanup
    /// leaves one obvious file to notice rather than a mystery.
    const NAME: &str = "frost-store-test.tmp.ron";

    #[test]
    fn what_is_kept_is_read_back() {
        save(NAME, "(\n    hill: 3,\n)\n").expect("the launch folder takes the write");
        assert_eq!(load(NAME).as_deref(), Some("(\n    hill: 3,\n)\n"));
        // And over: the second keeping stands, not the first.
        save(NAME, "(\n    hill: 4,\n)\n").expect("the write lands again");
        assert_eq!(load(NAME).as_deref(), Some("(\n    hill: 4,\n)\n"));
        std::fs::remove_file(NAME).expect("the test leaves no file");
    }

    #[test]
    fn a_name_never_kept_answers_nothing() {
        assert_eq!(load("frost-store-absent.tmp.ron"), None);
    }
}
