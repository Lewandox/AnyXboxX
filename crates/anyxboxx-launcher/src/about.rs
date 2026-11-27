//! The About line: the launcher's version and the git commit it was built from (embedded by
//! `build.rs`).

/// The launcher version (from `ANYXBOXX_LAUNCHER_VERSION`, the `launcher-v*` git tag, or the
/// crate's `Cargo.toml` version).
pub const VERSION: &str = env!("ANYXBOXX_LAUNCHER_VERSION");
/// The short commit hash of the build, `unknown` outside a git checkout.
pub const COMMIT: &str = env!("ANYXBOXX_GIT_COMMIT");

/// `AnyXboxX launcher 0.2.1, commit 1a2b3c4d`.
pub fn line() -> String {
    format!("AnyXboxX launcher {VERSION}, commit {COMMIT}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_about_line_names_the_version_and_the_commit() {
        assert!(!VERSION.is_empty() && !COMMIT.is_empty());
        assert_ne!(
            VERSION, "0.1.0",
            "launcher About version must track the launcher release, not the 0.1.0 workspace default"
        );
        let line = line();
        assert!(line.contains(VERSION) && line.contains(COMMIT));
        assert!(COMMIT == "unknown" || COMMIT.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
