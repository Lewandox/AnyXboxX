//! Bookkeeping of the pass over the view heap's buffer views for shaders
//! that pick them by a computed index (`draw::State::heap_buffers`): which
//! slots it looks at again, and the span of each host copy their views
//! read. Kept apart from D3D12 so that it can be tested alone.
//!
//! The pass used to look again at every buffer view of the heap (81,000 in
//! the world) every 16 passes, and to rebuild the spans of all host copies
//! from all views every pass. Now a slot is looked at again only when
//! - its buffer view was written or went (the heap's own set of changes);
//! - the view table entry its record names was freed (the title released
//!   the resource: `descriptor::forget`), which leaves the record as it was;
//! - its record in title memory is not the one seen at the last look (a
//!   record copied by the title itself, past the API); a compare of 12
//!   bytes per slot, no lock and no lookup;
//! - it found no host copy at its last look: every 16 passes, as before;
//! - every [`FULL_EVERY`] passes, all of them, as a safety net.
//!
//! The span of a copy is worked out again only when one of its slots went.
//! Off by default: the old pass (all slots every 16 passes, all spans every
//! pass) stays. `VICTORY_HEAP_PASS_DIRTY=1` turns the marked pass on.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::hash::{BuildHasher, Hasher};

/// Fast deterministic hasher for integer keys (`u32` slot indices and `usize`
/// host-copy pointers) in the heap pass's internal maps and sets.
#[derive(Clone, Default)]
struct IntBuildHasher;

struct IntHasher(u64);

impl BuildHasher for IntBuildHasher {
    type Hasher = IntHasher;

    fn build_hasher(&self) -> Self::Hasher {
        IntHasher(0)
    }
}

#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
    x ^= x >> 33;
    x = x.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    x ^ (x >> 33)
}

impl Hasher for IntHasher {
    fn finish(&self) -> u64 {
        mix64(self.0)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = self.0.rotate_left(5).wrapping_mul(0x517c_c1b7_2722_0a95) ^ u64::from(b);
        }
    }

    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(i)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn write_usize(&mut self, i: usize) {
        self.0 = (self.0.rotate_left(5) ^ (i as u64)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
}

/// Passes between two looks at every slot, besides those marked.
pub(crate) const FULL_EVERY: u64 = 256;
/// Passes between two looks at slots that found no host copy (and between
/// two looks at every slot with the old pass).
pub(crate) const ORPHANS_EVERY: u64 = 16;

/// Whether the old pass is used: unless `VICTORY_HEAP_PASS_DIRTY=1` asks for the marked one.
pub(crate) fn full_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var_os("VICTORY_HEAP_PASS_DIRTY").is_some_and(|v| v == "1"))
}

/// Passes between two scans of all kept slots while `VICTORY_HEAP_SCAN_ALT` asks for the slow scan.
pub(crate) const ALT_SCAN_EVERY: u64 = 16;
/// Seconds per phase of `VICTORY_HEAP_SCAN_ALT=alt`.
const ALT_PHASE_S: u64 = 20;

/// `VICTORY_HEAP_SCAN_ALT`: `1` scans every [`ALT_SCAN_EVERY`] passes all the time; `alt` does so
/// for twenty seconds, then scans every pass for twenty seconds, in turn (each change said in the
/// log). Unset or `0`: the scan interval stays `VICTORY_HEAP_SCAN_EVERY` (every pass by default).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AltMode {
    Off,
    Always,
    Toggle,
}

pub(crate) fn alt_mode() -> AltMode {
    static MODE: std::sync::OnceLock<AltMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_HEAP_SCAN_ALT").as_deref() {
        Ok("1") => AltMode::Always,
        Ok("alt") => AltMode::Toggle,
        _ => AltMode::Off,
    })
}

/// Whether the slow scan is on `elapsed_secs` after the start, in `mode`.
pub(crate) fn alt_slow_phase(mode: AltMode, elapsed_secs: u64) -> bool {
    match mode {
        AltMode::Off => false,
        AltMode::Always => true,
        AltMode::Toggle => (elapsed_secs / ALT_PHASE_S) % 2 == 1,
    }
}

/// The interval between two scans: `base` (the configured one) or, in the slow phase, the slower
/// of `base` and [`ALT_SCAN_EVERY`].
pub(crate) fn interval_for(base: u64, slow: bool) -> u64 {
    if slow { base.max(ALT_SCAN_EVERY) } else { base }
}

/// A slot's record in title memory: the tag and two words
/// (`descriptor::record_at`).
pub(crate) type Record = [u32; 3];

struct Slot {
    copy: usize,
    from: u64,
    to: u64,
    record: Record,
}

struct Copy<C> {
    copy: C,
    slots: HashSet<u32, IntBuildHasher>,
    from: u64,
    to: u64,
    /// A slot went or shrank off a boundary: the span is worked out again from those left.
    stale: bool,
}

/// Passes between two scans of all kept slots' raw records in title memory
/// for past-the-API copies (`VICTORY_HEAP_SCAN_EVERY`, default 1: every pass).
pub(crate) fn scan_every() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_HEAP_SCAN_EVERY")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1)
    })
}

/// What a pass looks at (see the module).
pub(crate) struct Picked<C> {
    passes: u64,
    /// Kept slots in slot order so the per-pass record scan reads title
    /// descriptor memory sequentially instead of in hash order.
    slots: BTreeMap<u32, Slot>,
    copies: HashMap<usize, Copy<C>, IntBuildHasher>,
    /// Slots that found no host copy, with the record seen, in slot order.
    orphans: BTreeMap<u32, Record>,
    /// View table entry -> slots whose record named it at their last look
    /// (some may name another by now: checked when used).
    by_entry: HashMap<u32, Vec<u32>, IntBuildHasher>,
    /// Table entries freed since the last pass; `None`: not known.
    forgotten: Option<Vec<u32>>,
    /// The old pass (the default; `VICTORY_HEAP_PASS_DIRTY=1` turns it off).
    full: bool,
    /// Interval for scanning all kept slots for past-the-API record changes.
    scan_every: u64,
    /// The slow scan of `VICTORY_HEAP_SCAN_ALT` was on at the last pass (logged when it changes).
    alt_slow: bool,
}

impl<C: Clone> Picked<C> {
    pub(crate) fn new(full: bool) -> Self {
        Self::with_scan_every(full, if cfg!(test) { 1 } else { scan_every() })
    }

    pub(crate) fn with_scan_every(full: bool, scan_every: u64) -> Self {
        Picked {
            passes: 0,
            slots: BTreeMap::new(),
            copies: HashMap::with_hasher(IntBuildHasher),
            orphans: BTreeMap::new(),
            by_entry: HashMap::with_hasher(IntBuildHasher),
            forgotten: None,
            full,
            scan_every: scan_every.max(1),
            alt_slow: false,
        }
    }

    /// View table entries were freed (`None`: some were, which is not
    /// known): their slots are looked at next pass. Every heap's pass is
    /// told, though one of them asks.
    pub(crate) fn note_forgotten(&mut self, entries: Option<&[u32]>) {
        match (entries, &mut self.forgotten) {
            (Some(new), Some(kept)) => kept.extend_from_slice(new),
            _ => self.forgotten = None,
        }
    }

    /// The slots this pass looks at again, in order. `changed`: the slots
    /// whose buffer view was written or went since the last pass (and the
    /// entries [`Picked::note_forgotten`] was told of: if it was told it
    /// does not know, all slots are looked at); `all`: every slot that holds a
    /// buffer view; `record`: a slot's record now; `entry`: the table
    /// entry a record names, if it names one.
    pub(crate) fn plan(
        &mut self,
        changed: Vec<u32>,
        all: impl FnOnce() -> Vec<u32>,
        record: impl Fn(u32) -> Option<Record>,
        entry: impl Fn(Record) -> Option<u32>,
    ) -> Vec<u32> {
        let pass = self.passes;
        self.passes += 1;
        let forgotten = self.forgotten.replace(Vec::new());
        let every = self.current_scan_every();
        if self.full {
            if pass.is_multiple_of(ORPHANS_EVERY) {
                self.clear();
                return all();
            }
            return changed;
        }
        let Some(forgotten) = forgotten else {
            self.clear();
            return all();
        };
        if pass.is_multiple_of(FULL_EVERY) {
            self.clear();
            return all();
        }
        let mut due = changed;
        for e in forgotten {
            if let Some(slots) = self.by_entry.remove(&e) {
                // Only those whose record still names the entry.
                due.extend(
                    slots
                        .into_iter()
                        .filter(|s| self.record_of(*s).is_some_and(|r| entry(r) == Some(e))),
                );
            }
        }
        let scan_records = pass.is_multiple_of(every);
        if scan_records {
            for (&slot, kept) in &self.slots {
                if record(slot) != Some(kept.record) {
                    due.push(slot);
                }
            }
        }
        let orphans_too = pass.is_multiple_of(ORPHANS_EVERY);
        if orphans_too || scan_records {
            for (&slot, &seen) in &self.orphans {
                if orphans_too || record(slot) != Some(seen) {
                    due.push(slot);
                }
            }
        }
        if due.len() > 1 {
            due.sort_unstable();
            due.dedup();
        }
        due
    }

    /// The scan interval now: the configured one, or the slow one in the slow phase of `VICTORY_HEAP_SCAN_ALT`.
    fn current_scan_every(&mut self) -> u64 {
        let mode = alt_mode();
        if mode == AltMode::Off {
            return self.scan_every;
        }
        static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        let elapsed = EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs();
        let slow = alt_slow_phase(mode, elapsed);
        if slow != self.alt_slow {
            self.alt_slow = slow;
            tracing::info!(
                "heap scan of every kept slot for past-the-API records (VICTORY_HEAP_SCAN_ALT): {}",
                if slow { "every 16 passes" } else { "every pass, as before" }
            );
        }
        interval_for(self.scan_every, slow)
    }

    fn record_of(&self, slot: u32) -> Option<Record> {
        self.slots
            .get(&slot)
            .map(|s| s.record)
            .or_else(|| self.orphans.get(&slot).copied())
    }

    fn clear(&mut self) {
        self.slots.clear();
        self.copies.clear();
        self.orphans.clear();
        self.by_entry.clear();
    }

    /// What slot `slot` held is let go (it is looked at now).
    pub(crate) fn forget(&mut self, slot: u32) {
        if self.slots.is_empty() && self.orphans.is_empty() {
            return;
        }
        self.orphans.remove(&slot);
        let Some(old) = self.slots.remove(&slot) else {
            return;
        };
        if let Some(copy) = self.copies.get_mut(&old.copy) {
            copy.slots.remove(&slot);
            if copy.slots.is_empty() {
                self.copies.remove(&old.copy);
            } else if old.from <= copy.from || old.to >= copy.to {
                copy.stale = true;
            }
        }
    }

    /// Slot `slot` (with `record`, naming table entry `entry`) reads title
    /// memory `[from, to)` through host copy `copy` (`key`: its identity);
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn keep(
        &mut self,
        slot: u32,
        key: usize,
        copy: &C,
        [from, to]: [u64; 2],
        record: Record,
        entry: Option<u32>,
    ) {
        // Fast path: updating a slot that still maps to the same host copy
        // avoids removing/re-inserting in `copy.slots` and avoids marking the
        // copy's span stale unless the slot shrank off a boundary.
        if let Some(old) = self.slots.get_mut(&slot)
            && old.copy == key
        {
            let (old_from, old_to) = (old.from, old.to);
            old.from = from;
            old.to = to;
            old.record = record;
            if let Some(c) = self.copies.get_mut(&key) {
                if (old_from <= c.from && from > old_from) || (old_to >= c.to && to < old_to) {
                    c.stale = true;
                }
                c.from = c.from.min(from);
                c.to = c.to.max(to);
            }
            // Always re-register `entry`: `plan()` drains forgotten entries from
            // `by_entry`, so a slot re-checked with the same record still needs
            // to be notified if that entry is freed again later.
            self.note_entry(slot, entry);
            return;
        }
        self.forget(slot);
        self.note_entry(slot, entry);
        self.slots.insert(slot, Slot { copy: key, from, to, record });
        let c = self.copies.entry(key).or_insert_with(|| Copy {
            copy: copy.clone(),
            slots: HashSet::with_hasher(IntBuildHasher),
            from,
            to,
            stale: false,
        });
        c.slots.insert(slot);
        c.from = c.from.min(from);
        c.to = c.to.max(to);
    }

    /// Slot `slot` holds a buffer view whose memory has no host copy yet.
    pub(crate) fn orphan(&mut self, slot: u32, record: Record, entry: Option<u32>) {
        if self.orphans.get(&slot) == Some(&record) {
            self.note_entry(slot, entry);
            return;
        }
        self.forget(slot);
        self.note_entry(slot, entry);
        self.orphans.insert(slot, record);
    }

    fn note_entry(&mut self, slot: u32, entry: Option<u32>) {
        // Only kept with the incremental pass (the old one does not use it).
        if let Some(e) = entry
            && !self.full
        {
            // (Maybe twice: the pass's set of due slots takes it once.)
            self.by_entry.entry(e).or_default().push(slot);
        }
    }

    /// The views kept.
    pub(crate) fn views(&self) -> usize {
        self.slots.len()
    }

    /// Per host copy, the span its views read: `(copy, from, to)`.
    pub(crate) fn spans(&mut self) -> Vec<(C, u64, u64)> {
        let slots = &self.slots;
        let mut out = Vec::with_capacity(self.copies.len());
        for c in self.copies.values_mut() {
            if c.stale {
                c.stale = false;
                let (mut from, mut to) = (u64::MAX, 0);
                for s in c.slots.iter().filter_map(|s| slots.get(s)) {
                    from = from.min(s.from);
                    to = to.max(s.to);
                }
                (c.from, c.to) = (from, to);
            }
            out.push((c.copy.clone(), c.from, c.to));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A heap of slots with records, as title memory holds them.
    struct Heap {
        records: HashMap<u32, Record>,
    }

    impl Heap {
        fn record(&self, slot: u32) -> Option<Record> {
            self.records.get(&slot).copied()
        }
    }

    fn entry(r: Record) -> Option<u32> {
        Some(r[1])
    }

    /// One pass: plans, then "looks" at each due slot: a slot reads 16
    /// bytes from `slot * 16` of copy `slot / 4`.
    fn pass(p: &mut Picked<u32>, heap: &Heap, changed: Vec<u32>, forgotten: Vec<u32>) -> Vec<u32> {
        p.note_forgotten(Some(&forgotten));
        let due = p.plan(
            changed,
            || {
                let mut all: Vec<u32> = heap.records.keys().copied().collect();
                all.sort_unstable();
                all
            },
            |s| heap.record(s),
            entry,
        );
        for &slot in &due {
            let Some(record) = heap.record(slot) else {
                p.forget(slot);
                continue;
            };
            let copy = slot / 4;
            let from = u64::from(slot) * 16;
            p.keep(slot, copy as usize, &copy, [from, from + 16], record, entry(record));
        }
        due
    }

    fn heap(n: u32) -> Heap {
        Heap {
            records: (0..n).map(|s| (s, [1, 100 + s, 1])).collect(),
        }
    }

    #[test]
    fn only_slots_changed_since_the_last_pass_are_looked_at_again() {
        let mut heap = heap(64);
        let mut p = Picked::new(false);
        // The first pass looks at all of them.
        assert_eq!(pass(&mut p, &heap, vec![], vec![]).len(), 64);
        assert_eq!(p.views(), 64);
        // Nothing changed: nothing is looked at.
        assert!(pass(&mut p, &heap, vec![], vec![]).is_empty());
        // A view written through the API.
        heap.records.insert(5, [1, 500, 1]);
        assert_eq!(pass(&mut p, &heap, vec![5], vec![]), vec![5]);
        // A record copied past the API: found by its words alone.
        heap.records.insert(9, [1, 900, 1]);
        assert_eq!(pass(&mut p, &heap, vec![], vec![]), vec![9]);
        // The resource of entry 117 (slot 17) released: the record is as
        // it was, the slot is looked at all the same; slot 5 no longer
        // names its old entry 105 and is not.
        assert_eq!(pass(&mut p, &heap, vec![], vec![117, 105]), vec![17]);
        // And the next pass leaves them alone again.
        assert!(pass(&mut p, &heap, vec![], vec![]).is_empty());
    }

    #[test]
    fn a_slot_that_went_shrinks_its_copy_span() {
        let mut heap = heap(8);
        let mut p = Picked::new(false);
        pass(&mut p, &heap, vec![], vec![]);
        let span = |p: &mut Picked<u32>, copy: u32| p.spans().into_iter().find(|s| s.0 == copy).map(|s| (s.1, s.2));
        assert_eq!(span(&mut p, 1), Some((64, 128)));
        heap.records.remove(&7);
        pass(&mut p, &heap, vec![7], vec![]);
        assert_eq!(span(&mut p, 1), Some((64, 112)));
        for s in 4..7 {
            heap.records.remove(&s);
        }
        pass(&mut p, &heap, vec![4, 5, 6], vec![]);
        assert_eq!(span(&mut p, 1), None);
        assert_eq!(span(&mut p, 0), Some((0, 64)));
    }

    #[test]
    fn orphans_are_tried_again_every_16_passes_and_all_slots_now_and_then() {
        let heap = heap(4);
        let mut p: Picked<u32> = Picked::new(false);
        let _ = p.plan(vec![], || vec![0, 1, 2, 3], |s| heap.record(s), entry);
        for s in 0..3 {
            p.keep(s, 0, &0, [0, 16], heap.record(s).unwrap(), None);
        }
        p.orphan(3, heap.record(3).unwrap(), None);
        for n in 1..FULL_EVERY {
            let due = p.plan(vec![], || vec![0, 1, 2, 3], |s| heap.record(s), entry);
            let want: Vec<u32> = if n.is_multiple_of(ORPHANS_EVERY) { vec![3] } else { vec![] };
            assert_eq!(due, want, "pass {n}");
        }
        let due = p.plan(vec![], || vec![0, 1, 2, 3], |s| heap.record(s), entry);
        assert_eq!(due, vec![0, 1, 2, 3]);
    }

    #[test]
    fn the_old_pass_looks_at_all_every_16_passes() {
        let heap = heap(4);
        let mut p: Picked<u32> = Picked::new(true);
        for n in 0..40u64 {
            let due = p.plan(vec![2], || vec![0, 1, 2, 3], |s| heap.record(s), entry);
            let want: Vec<u32> = if n.is_multiple_of(ORPHANS_EVERY) { vec![0, 1, 2, 3] } else { vec![2] };
            assert_eq!(due, want, "pass {n}");
        }
    }

    #[test]
    fn the_alt_scan_is_slow_only_in_its_slow_phase() {
        assert!(!alt_slow_phase(AltMode::Off, 0));
        assert!(!alt_slow_phase(AltMode::Off, 1000));
        assert!(alt_slow_phase(AltMode::Always, 0));
        assert!(!alt_slow_phase(AltMode::Toggle, 19));
        assert!(alt_slow_phase(AltMode::Toggle, 20));
        assert!(alt_slow_phase(AltMode::Toggle, 39));
        assert!(!alt_slow_phase(AltMode::Toggle, 40));
    }

    #[test]
    fn the_slow_phase_scans_every_16_passes_and_the_default_stays() {
        assert_eq!(interval_for(1, false), 1);
        assert_eq!(interval_for(1, true), ALT_SCAN_EVERY);
        assert_eq!(interval_for(32, true), 32);
    }

    #[test]
    fn in_place_keep_avoids_stale_unless_shrinking_a_boundary() {
        let mut p: Picked<u32> = Picked::new(false);
        p.keep(0, 1, &1, [100, 200], [1, 10, 1], Some(10));
        p.keep(1, 1, &1, [150, 250], [1, 11, 1], Some(11));
        assert!(!p.copies[&1].stale);
        assert_eq!(p.spans(), vec![(1, 100, 250)]);

        // Re-keeping an unchanged slot or expanding a boundary stays non-stale.
        p.keep(0, 1, &1, [100, 200], [1, 10, 1], Some(10));
        assert!(!p.copies[&1].stale);
        p.keep(0, 1, &1, [80, 200], [1, 10, 2], Some(10));
        assert!(!p.copies[&1].stale);
        assert_eq!(p.spans(), vec![(1, 80, 250)]);

        // Shrinking a boundary marks stale and recomputes exact span.
        p.keep(0, 1, &1, [120, 200], [1, 10, 3], Some(10));
        assert!(p.copies[&1].stale);
        assert_eq!(p.spans(), vec![(1, 120, 250)]);
        assert!(!p.copies[&1].stale);

        // Moving a slot to another host copy shrinks the old copy and creates the new one.
        p.keep(0, 2, &2, [0, 64], [1, 20, 1], Some(20));
        let mut spans = p.spans();
        spans.sort_unstable();
        assert_eq!(spans, vec![(1, 150, 250), (2, 0, 64)]);
    }

    #[test]
    fn full_and_incremental_modes_report_identical_spans() {
        let mut full: Picked<u32> = Picked::new(true);
        let mut inc: Picked<u32> = Picked::new(false);
        let sorted_spans = |p: &mut Picked<u32>| {
            let mut s = p.spans();
            s.sort_unstable();
            s
        };
        for p in [&mut full, &mut inc] {
            p.keep(0, 10, &10, [64, 128], [1, 1, 1], Some(1));
            p.keep(1, 10, &10, [128, 256], [1, 2, 1], Some(2));
            p.keep(2, 20, &20, [0, 32], [1, 3, 1], Some(3));
        }
        assert_eq!(sorted_spans(&mut full), sorted_spans(&mut inc));
        assert_eq!(sorted_spans(&mut full), vec![(10, 64, 256), (20, 0, 32)]);

        // Shrink upper boundary, remove lower boundary, and switch copy.
        for p in [&mut full, &mut inc] {
            p.keep(1, 10, &10, [128, 192], [1, 2, 2], Some(2));
            p.forget(0);
            p.keep(2, 10, &10, [16, 48], [1, 3, 2], Some(3));
        }
        assert_eq!(sorted_spans(&mut full), sorted_spans(&mut inc));
        assert_eq!(sorted_spans(&mut full), vec![(10, 16, 192)]);
    }

    #[test]
    fn plan_deduplicates_and_sorts_due_slots_across_all_sources() {
        let mut heap = heap(8);
        let mut p = Picked::new(false);
        pass(&mut p, &heap, vec![], vec![]);
        // Slot 3 changes in title memory AND is passed in `changed` twice AND its entry is forgotten.
        heap.records.insert(3, [1, 999, 2]);
        let due = pass(&mut p, &heap, vec![5, 3, 3, 1], vec![103, 101]);
        assert_eq!(due, vec![1, 3, 5]);
    }

    #[test]
    fn forgotten_entry_re_registers_on_repeated_keep_and_orphan_with_same_record() {
        let heap = heap(4);
        let mut p: Picked<u32> = Picked::new(false);
        let _ = p.plan(vec![], || vec![0, 1], |s| heap.record(s), entry);
        let rec0 = heap.record(0).unwrap();
        let rec1 = heap.record(1).unwrap();
        p.keep(0, 10, &10, [0, 16], rec0, entry(rec0));
        p.orphan(1, rec1, entry(rec1));

        // First forget drains `by_entry` for both entries.
        p.note_forgotten(Some(&[100, 101]));
        let due1 = p.plan(vec![], || vec![0, 1], |s| heap.record(s), entry);
        assert_eq!(due1, vec![0, 1]);
        // Re-keeping and re-orphaning with the exact same records must re-register `by_entry`.
        p.keep(0, 10, &10, [0, 16], rec0, entry(rec0));
        p.orphan(1, rec1, entry(rec1));

        // Second forget of the same entries must still schedule both slots.
        p.note_forgotten(Some(&[100, 101]));
        let due2 = p.plan(vec![], || vec![0, 1], |s| heap.record(s), entry);
        assert_eq!(due2, vec![0, 1]);
    }

    #[test]
    fn int_hasher_avalanche_mixes_aligned_pointer_low_bits() {
        use std::hash::Hash;
        let bh = IntBuildHasher;
        let mut low_buckets = HashSet::new();
        for i in 0..4096usize {
            let aligned_ptr = 0x0000_0200_0000_0000usize + (i << 6); // 64-byte aligned
            let mut h = bh.build_hasher();
            aligned_ptr.hash(&mut h);
            low_buckets.insert(h.finish() & 0xfff);
        }
        // With 4096 uniform hashes into 4096 buckets, expected distinct buckets is ~2590;
        // unmixed aligned pointers would hit only 64 buckets (`<< 6` leaves 6 zero low bits).
        assert!(
            low_buckets.len() > 2000,
            "aligned pointers collided in low hash bits: only {} distinct buckets",
            low_buckets.len()
        );
    }
}
