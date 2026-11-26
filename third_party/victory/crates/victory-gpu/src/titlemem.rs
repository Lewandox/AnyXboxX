//! Where the title's own memory is, for the resident line (`ramline`): ranges the
//! runtime tells us about when it allocates, maps, unmaps or frees for the title.
//! Not per-frame: a lock and a map insert per call, and only large heap blocks.

use std::collections::BTreeMap;
use std::sync::Mutex;

/// `XMemVirtualAlloc` reservations (the title frees them itself, so entries can go stale).
pub const VIRTUAL: u8 = 1;
/// Views of physical pages (pool views, page mappings).
pub const VIEW: u8 = 2;
/// Large `XMemAlloc` heap blocks (small ones are not listed).
pub const HEAP: u8 = 3;
/// The title's exe and DLLs.
pub const IMAGE: u8 = 4;

static RANGES: Mutex<BTreeMap<usize, (usize, u8)>> = Mutex::new(BTreeMap::new());

fn ranges() -> std::sync::MutexGuard<'static, BTreeMap<usize, (usize, u8)>> {
    RANGES.lock().unwrap_or_else(|e| e.into_inner())
}

/// `[base, base + size)` is title memory of `kind` (replaces a range at `base`
/// and drops any stale same-kind sub-ranges starting inside `[base, base + size)`).
pub fn title_range(base: usize, size: usize, kind: u8) {
    let mut r = ranges();
    if size == 0 {
        r.remove(&base);
        return;
    }
    let end = base.saturating_add(size);
    let stale: Vec<usize> = r
        .range(base..end)
        .filter(|&(&k, &(_, existing_kind))| k != base && existing_kind == kind)
        .map(|(&k, _)| k)
        .collect();
    for k in stale {
        r.remove(&k);
    }
    r.insert(base, (size, kind));
}

/// The range that began at `base` is gone.
pub fn title_range_free(base: usize) {
    ranges().remove(&base);
}

/// A copy of the ranges as (base, size, kind), by base. The lock is not held after.
pub(crate) fn snapshot() -> Vec<(usize, usize, u8)> {
    ranges().iter().map(|(&b, &(s, k))| (b, s, k)).collect()
}

/// Forgets a `VIRTUAL` range the title freed behind our back (its base is no longer an allocation base).
pub(crate) fn forget_if_stale(base: usize) {
    let mut r = ranges();
    if r.get(&base).is_some_and(|e| e.1 == VIRTUAL) {
        r.remove(&base);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_range_replaces_same_kind_subranges_and_preserves_other_kinds() {
        let base = 0x7000_0000_0000_usize;
        title_range(base, 0x2000, VIRTUAL);
        title_range(base + 0x1000, 0x1000, VIRTUAL);
        title_range(base + 0x1000, 0x1000, VIEW);

        // Replacing [base, base + 0x4000) as VIRTUAL drops stale VIRTUAL sub-ranges,
        // while leaving the VIEW entry at base + 0x1000 intact? Note `base + 0x1000`
        // was overwritten by VIEW above, so add a distinct VIRTUAL sub-range at +0x2000.
        title_range(base + 0x2000, 0x1000, VIRTUAL);
        title_range(base, 0x4000, VIRTUAL);

        let snap = snapshot();
        assert!(snap.contains(&(base, 0x4000, VIRTUAL)));
        assert!(snap.contains(&(base + 0x1000, 0x1000, VIEW)));
        assert!(!snap.iter().any(|&(b, _, _)| b == base + 0x2000));

        // forget_if_stale only removes VIRTUAL entries, not VIEW entries.
        forget_if_stale(base + 0x1000);
        assert!(snapshot().contains(&(base + 0x1000, 0x1000, VIEW)));
        forget_if_stale(base);
        assert!(!snapshot().iter().any(|&(b, _, _)| b == base));

        // Zero-sized insert removes an existing entry at that base.
        title_range(base + 0x1000, 0, VIEW);
        assert!(!snapshot().iter().any(|&(b, _, _)| b == base + 0x1000));
    }
}
