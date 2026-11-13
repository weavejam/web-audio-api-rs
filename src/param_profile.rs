//! Opt-in offline attribution. Build with `param-profile`, then set
//! `TUTTI_PROFILE=1`. Timings are inclusive and intended for serial renders.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Instant;

const NAMES: [&str; 13] = [
    "inlineVisit",
    "inlineVisited",
    "inlineAlreadySettled",
    "ownerCleanSkipped",
    "ordinaryParam",
    "biquad",
    "biquadSetup",
    "convolver",
    "gain",
    "quanta",
    "paramEvaluate",
    "paramIdleSkip",
    "ownerVisit",
];
static NANOS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];
static COUNTS: [AtomicU64; 13] = [const { AtomicU64::new(0) }; 13];

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("TUTTI_PROFILE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

pub(crate) fn count(slot: usize, n: usize) {
    if enabled() {
        COUNTS[slot].fetch_add(n as u64, Ordering::Relaxed);
    }
}

pub(crate) struct Timer(usize, Option<Instant>);

impl Timer {
    pub(crate) fn new(slot: usize) -> Self {
        Self(slot, enabled().then(Instant::now))
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        if let Some(start) = self.1 {
            NANOS[self.0].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
            COUNTS[self.0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Snapshot counters after rendering, so reporting I/O is outside wall timing.
pub fn report_json() -> String {
    let rows: Vec<_> = NAMES
        .iter()
        .enumerate()
        .map(|(i, name)| {
            format!(
                "\"{name}\": {{\"nanos\": {}, \"count\": {}}}",
                NANOS[i].load(Ordering::Relaxed),
                COUNTS[i].load(Ordering::Relaxed)
            )
        })
        .collect();
    format!("{{{}}}", rows.join(","))
}

/// Clear the process-wide counters between renders, with no render running.
pub fn reset() {
    for counter in NANOS.iter().chain(COUNTS.iter()) {
        counter.store(0, Ordering::Relaxed);
    }
}
