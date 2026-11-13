//! Realtime scheduling for the threads that render audio (native only).
//!
//! Every thread on the audio critical path runs in a realtime / pro-audio
//! scheduling class while it renders:
//!
//! - the parallel executor's persistent render workers (for their lifetime);
//! - the online render thread, i.e. the thread that calls
//!   [`RenderThread::render`](super::RenderThread::render) (the device callback
//!   thread, or the `none` backend's timer thread). On Windows it is registered
//!   with MMCSS on its first callback and reverted when that thread exits.
//!   CoreAudio and AAudio already run their callback threads at realtime
//!   priority, so on Apple platforms and Android it is left as it is;
//! - on Windows, the thread that runs `OfflineAudioContext::start_rendering_sync`
//!   while the render uses the worker pool: it is worker 0, and the promoted
//!   workers wait on its partitions, so a preempted caller stalls them all. A
//!   serial offline render keeps the caller's scheduling.
//!
//! Promotion is best-effort and never changes the rendered output. Every
//! registration is undone on the thread that made it ([`RtPromotion`]'s
//! `Drop`).
//!
//! Windows allows at most 32 concurrent MMCSS registrations per process (the
//! 33rd fails with `ERROR_TOO_MANY_THREADS`; MMCSS releases a registration when
//! its thread exits). Past that limit a thread falls back to
//! `THREAD_PRIORITY_TIME_CRITICAL` ([`AudioThreadPriority::Elevated`]), which
//! still runs ahead of every normal-priority thread of the process's class.
//!
//! Denormals: worker 0 renders inside `no_denormals` (FTZ + DAZ on x86, FZ on
//! aarch64); the pool workers do the same for their whole lifetime, so a node
//! sees the same floating-point environment on every thread.

use std::marker::PhantomData;

/// Outcome of trying to raise a thread to a realtime scheduling class.
///
/// Which variants are reachable is platform-dependent (e.g. Windows MMCSS never
/// yields `Unsupported`), so some are inert on any single target.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AudioThreadPriority {
    /// The thread now runs in a realtime / pro-audio scheduling class.
    Promoted,
    /// Windows: MMCSS refused (e.g. the per-process registration limit), the
    /// thread runs at `THREAD_PRIORITY_TIME_CRITICAL` instead.
    Elevated,
    /// No supported mechanism is compiled in for this platform.
    Unsupported,
    /// The OS refused the elevation (e.g. no `CAP_SYS_NICE` / `RLIMIT_RTPRIO`
    /// on Linux). The thread keeps running at normal priority.
    Denied,
}

/// A realtime promotion of the current thread. Dropping it (on the same thread;
/// it is neither `Send` nor `Sync`) restores the scheduling the thread had
/// before and, on Windows, ends the MMCSS registration.
pub(crate) struct RtPromotion {
    saved: imp::Saved,
    _not_send: PhantomData<*const ()>,
}

impl Drop for RtPromotion {
    fn drop(&mut self) {
        imp::restore(&self.saved);
    }
}

/// Raise the *current* thread to a realtime / pro-audio scheduling class.
/// Returns the outcome and, unless nothing changed, the guard that undoes it.
pub(crate) fn promote_current_thread() -> (AudioThreadPriority, Option<RtPromotion>) {
    match imp::promote() {
        Ok((saved, outcome)) => (
            outcome,
            Some(RtPromotion {
                saved,
                _not_send: PhantomData,
            }),
        ),
        Err(outcome) => (outcome, None),
    }
}

/// Called by the online render thread at the start of every callback: on
/// Windows, registers the calling thread with MMCSS "Pro Audio" once (the
/// registration lives in a thread local and is reverted when the thread exits).
/// One thread-local read per callback after that. A no-op elsewhere: CoreAudio
/// and AAudio callback threads are already realtime, and on Linux promoting the
/// ALSA / PipeWire thread is cpal's `realtime` feature.
#[inline]
pub(crate) fn ensure_callback_thread_promoted() {
    #[cfg(target_os = "windows")]
    {
        use std::cell::RefCell;
        thread_local! {
            // None: not tried yet on this thread; Some(_): tried (the guard if
            // anything changed).
            static CALLBACK_PROMOTION: RefCell<Option<Option<RtPromotion>>> =
                const { RefCell::new(None) };
        }
        // `try_with`: never panic on the render path, even during thread
        // teardown.
        let _ = CALLBACK_PROMOTION.try_with(|slot| {
            if let Ok(mut slot) = slot.try_borrow_mut() {
                if slot.is_none() {
                    let (outcome, guard) = promote_current_thread();
                    if outcome != AudioThreadPriority::Promoted {
                        log::debug!("render thread: realtime priority not acquired ({outcome:?})");
                    }
                    *slot = Some(guard);
                }
            }
        });
    }
}

/// Promotion of the calling thread for the duration of an offline render that
/// uses the worker pool (Windows only, see the module docs). Dropping the guard
/// restores the caller's scheduling.
pub(crate) fn promote_offline_render_thread(uses_worker_pool: bool) -> Option<RtPromotion> {
    #[cfg(target_os = "windows")]
    {
        if uses_worker_pool {
            return promote_current_thread().1;
        }
        None
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = uses_worker_pool;
        None
    }
}

#[cfg(target_os = "windows")]
mod imp {
    use super::AudioThreadPriority;
    use std::ffi::c_void;

    // Multimedia Class Scheduler Service (MMCSS): registering a thread under the
    // "Pro Audio" task gives it glitch-resistant realtime scheduling. avrt is
    // not linked by default, so name it explicitly.
    #[link(name = "avrt")]
    extern "system" {
        fn AvSetMmThreadCharacteristicsW(
            task_name: *const u16,
            task_index: *mut u32,
        ) -> *mut c_void;
        fn AvSetMmThreadPriority(handle: *mut c_void, priority: i32) -> i32;
        fn AvRevertMmThreadCharacteristics(handle: *mut c_void) -> i32;
    }
    // kernel32 is linked by std; declare the few functions we use.
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut c_void;
        fn SetThreadPriority(thread: *mut c_void, priority: i32) -> i32;
        fn GetThreadPriority(thread: *mut c_void) -> i32;
        #[cfg(test)]
        fn GetLastError() -> u32;
    }

    const THREAD_PRIORITY_TIME_CRITICAL: i32 = 15;
    const THREAD_PRIORITY_ERROR_RETURN: i32 = 0x7fff_ffff;
    /// `AVRT_PRIORITY_CRITICAL`: the highest priority within the MMCSS task.
    const AVRT_PRIORITY_CRITICAL: i32 = 2;

    pub(crate) struct Saved {
        /// The MMCSS registration (null: the TIME_CRITICAL fallback).
        mmcss: *mut c_void,
        previous_priority: i32,
    }

    fn register() -> *mut c_void {
        // "Pro Audio" as a NUL-terminated UTF-16 string.
        let task: Vec<u16> = "Pro Audio\0".encode_utf16().collect();
        let mut task_index: u32 = 0;
        // SAFETY: `task` is a valid NUL-terminated UTF-16 buffer and
        // `task_index` a valid out pointer for the duration of the call.
        unsafe { AvSetMmThreadCharacteristicsW(task.as_ptr(), &mut task_index) }
    }

    pub(super) fn promote() -> Result<(Saved, AudioThreadPriority), AudioThreadPriority> {
        // SAFETY: pseudo-handle valid for the calling thread.
        let previous_priority = unsafe { GetThreadPriority(GetCurrentThread()) };
        let mmcss = register();
        // SAFETY: `mmcss` (when non-null) is the live registration handle of
        // this thread; `GetCurrentThread` returns a pseudo-handle valid for it.
        unsafe {
            if !mmcss.is_null() {
                AvSetMmThreadPriority(mmcss, AVRT_PRIORITY_CRITICAL);
            }
            // Also pin an explicit, queryable priority: what the thread runs at
            // whenever MMCSS is not boosting it, and the fallback when MMCSS
            // refused (the per-process registration limit).
            if SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL) == 0
                && mmcss.is_null()
            {
                return Err(AudioThreadPriority::Denied);
            }
        }
        let outcome = if mmcss.is_null() {
            AudioThreadPriority::Elevated
        } else {
            AudioThreadPriority::Promoted
        };
        Ok((
            Saved {
                mmcss,
                previous_priority,
            },
            outcome,
        ))
    }

    pub(super) fn restore(saved: &Saved) {
        // SAFETY: `saved.mmcss` was returned by AvSetMmThreadCharacteristicsW on
        // this thread (the guard is !Send) and is reverted exactly once.
        unsafe {
            if !saved.mmcss.is_null() {
                AvRevertMmThreadCharacteristics(saved.mmcss);
            }
            if saved.previous_priority != THREAD_PRIORITY_ERROR_RETURN {
                SetThreadPriority(GetCurrentThread(), saved.previous_priority);
            }
        }
    }

    #[cfg(test)]
    pub(super) fn current_thread_priority() -> i32 {
        // SAFETY: pseudo-handle valid for the calling thread.
        unsafe { GetThreadPriority(GetCurrentThread()) }
    }

    #[cfg(test)]
    pub(super) fn current_thread_is_elevated() -> bool {
        current_thread_priority() == THREAD_PRIORITY_TIME_CRITICAL
    }

    /// Register without the priority calls (diagnostics). Returns the handle
    /// (null on failure) and `GetLastError` on failure.
    #[cfg(test)]
    pub(super) fn raw_register() -> (*mut c_void, u32) {
        let h = register();
        // SAFETY: no preconditions.
        let err = if h.is_null() {
            unsafe { GetLastError() }
        } else {
            0
        };
        (h, err)
    }

    #[cfg(test)]
    pub(super) fn raw_revert(h: *mut c_void) {
        // SAFETY: `h` came from `raw_register` on this thread.
        unsafe {
            AvRevertMmThreadCharacteristics(h);
        }
    }
}

#[cfg(unix)]
mod imp {
    use super::AudioThreadPriority;

    pub(crate) struct Saved {
        policy: i32,
        param: libc::sched_param,
    }

    pub(super) fn promote() -> Result<(Saved, AudioThreadPriority), AudioThreadPriority> {
        let mut policy = 0i32;
        // SAFETY: `sched_param` is a plain C struct; all-zero is a valid value.
        let mut previous: libc::sched_param = unsafe { std::mem::zeroed() };
        // SAFETY: both out pointers are valid for the call.
        if unsafe { libc::pthread_getschedparam(libc::pthread_self(), &mut policy, &mut previous) }
            != 0
        {
            return Err(AudioThreadPriority::Unsupported);
        }
        // SAFETY: both query the static scheduling policy; no pointers involved.
        let min = unsafe { libc::sched_get_priority_min(libc::SCHED_FIFO) };
        let max = unsafe { libc::sched_get_priority_max(libc::SCHED_FIFO) };
        if min < 0 || max < 0 {
            return Err(AudioThreadPriority::Unsupported);
        }
        // A middling realtime priority: above ordinary work, below the most
        // critical kernel/driver threads. Zero-init the struct (it has
        // platform-private padding on macOS, so a field literal won't compile).
        // SAFETY: `sched_param` is a plain C struct; all-zero is a valid value.
        let mut param: libc::sched_param = unsafe { std::mem::zeroed() };
        param.sched_priority = min + (max - min) / 2;
        // SAFETY: `param` outlives the call; `pthread_self` is always valid.
        let rc =
            unsafe { libc::pthread_setschedparam(libc::pthread_self(), libc::SCHED_FIFO, &param) };
        if rc == 0 {
            Ok((
                Saved {
                    policy,
                    param: previous,
                },
                AudioThreadPriority::Promoted,
            ))
        } else {
            Err(AudioThreadPriority::Denied)
        }
    }

    pub(super) fn restore(saved: &Saved) {
        // SAFETY: `saved.param` outlives the call; `pthread_self` is valid.
        unsafe {
            libc::pthread_setschedparam(libc::pthread_self(), saved.policy, &saved.param);
        }
    }

    #[cfg(test)]
    pub(super) fn current_thread_is_elevated() -> bool {
        let mut policy = 0i32;
        // SAFETY: `sched_param` is a plain C struct; all-zero is a valid value.
        let mut param: libc::sched_param = unsafe { std::mem::zeroed() };
        // SAFETY: both out pointers are valid for the call.
        let rc =
            unsafe { libc::pthread_getschedparam(libc::pthread_self(), &mut policy, &mut param) };
        rc == 0 && policy == libc::SCHED_FIFO
    }
}

#[cfg(all(not(unix), not(target_os = "windows")))]
mod imp {
    use super::AudioThreadPriority;

    pub(crate) struct Saved;

    pub(super) fn promote() -> Result<(Saved, AudioThreadPriority), AudioThreadPriority> {
        Err(AudioThreadPriority::Unsupported)
    }

    pub(super) fn restore(_: &Saved) {}

    #[cfg(test)]
    pub(super) fn current_thread_is_elevated() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::{imp, promote_current_thread, AudioThreadPriority};

    // The realtime-promotion mechanism must return a known outcome without
    // panicking, and when it reports an elevation the OS must actually confirm
    // the elevated scheduling state for that thread; dropping the guard must
    // give the thread its previous scheduling back. Runs on a fresh thread so it
    // never leaves the test runner at realtime priority.
    //
    // On Windows MMCSS registration needs no privilege (`Promoted`, or
    // `Elevated` while other tests hold the process's 32 registrations). On
    // Linux CI SCHED_FIFO is typically `EPERM` → `Denied`, which is an accepted
    // outcome (the pool still runs).
    #[test]
    fn promote_elevates_or_degrades_gracefully_and_restores() {
        let outcome = std::thread::spawn(|| {
            let before = imp::current_thread_is_elevated();
            let (result, guard) = promote_current_thread();
            if matches!(
                result,
                AudioThreadPriority::Promoted | AudioThreadPriority::Elevated
            ) {
                assert!(
                    imp::current_thread_is_elevated(),
                    "reported {result:?} but the OS does not confirm an elevated class"
                );
                drop(guard);
                assert_eq!(
                    imp::current_thread_is_elevated(),
                    before,
                    "dropping the promotion must restore the previous scheduling"
                );
            } else {
                assert!(guard.is_none());
            }
            result
        })
        .join()
        .expect("promotion test thread panicked");

        // Record the environment's outcome to ease debugging of CI scheduling.
        eprintln!("audio-thread priority outcome on this host: {outcome:?}");
        #[cfg(target_os = "windows")]
        assert_ne!(outcome, AudioThreadPriority::Denied);
    }

    // MMCSS registrations must not pile up: many short-lived threads that each
    // promote and exit (a pool per context), and one thread promoting and
    // reverting over and over (offline renders on one caller thread), must keep
    // getting "Pro Audio". The process may hold at most 32 registrations, so
    // a leak would show as `Elevated` after a few dozen rounds. Ignored by
    // default: other tests running at the same time hold registrations too
    // (`cargo test --release --lib rt_thread -- --ignored` runs it alone).
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore]
    fn promotion_does_not_leak_mmcss_registrations() {
        for i in 0..2_000 {
            let promoted = std::thread::spawn(|| promote_current_thread().0)
                .join()
                .unwrap();
            assert_eq!(promoted, AudioThreadPriority::Promoted, "thread {i}");
        }
        std::thread::spawn(|| {
            let before = imp::current_thread_priority();
            for i in 0..2_000 {
                let (outcome, guard) = promote_current_thread();
                assert_eq!(outcome, AudioThreadPriority::Promoted, "round {i}");
                drop(guard);
            }
            assert_eq!(imp::current_thread_priority(), before);
        })
        .join()
        .unwrap();
    }

    // Past the per-process MMCSS limit a thread still gets TIME_CRITICAL, and
    // its guard restores the previous priority. Ignored for the same reason as
    // the leak test.
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore]
    fn past_the_mmcss_limit_threads_fall_back_to_time_critical() {
        use std::sync::{Arc, Barrier};
        let n = 40;
        let hold = Arc::new(Barrier::new(n + 1));
        let (tx, rx) = std::sync::mpsc::channel();
        let threads: Vec<_> = (0..n)
            .map(|_| {
                let hold = Arc::clone(&hold);
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let before = imp::current_thread_priority();
                    let (outcome, guard) = promote_current_thread();
                    tx.send((outcome, imp::current_thread_is_elevated()))
                        .unwrap();
                    hold.wait();
                    drop(guard);
                    assert_eq!(imp::current_thread_priority(), before);
                })
            })
            .collect();
        let results: Vec<_> = (0..n).map(|_| rx.recv().unwrap()).collect();
        hold.wait();
        for t in threads {
            t.join().unwrap();
        }
        let promoted = results
            .iter()
            .filter(|r| r.0 == AudioThreadPriority::Promoted)
            .count();
        let elevated = results
            .iter()
            .filter(|r| r.0 == AudioThreadPriority::Elevated)
            .count();
        assert!(results.iter().all(|r| r.1), "every thread runs elevated");
        assert_eq!(promoted + elevated, n);
        assert!(
            elevated >= n - 32,
            "{promoted} promoted, {elevated} elevated"
        );
    }

    // Diagnostic (run with --ignored --nocapture): how MMCSS behaves when a
    // thread registers without reverting, and how many registrations one
    // process can hold at once.
    #[cfg(target_os = "windows")]
    #[test]
    #[ignore]
    fn mmcss_registration_probe() {
        // 1. Threads that register and exit without reverting.
        let mut failed_exit = 0;
        let mut first_err_exit = 0;
        for _ in 0..5_000 {
            let (ok, err) = std::thread::spawn(|| {
                let (h, err) = imp::raw_register();
                (!h.is_null(), err)
            })
            .join()
            .unwrap();
            if !ok {
                failed_exit += 1;
                if first_err_exit == 0 {
                    first_err_exit = err;
                }
            }
        }
        // 2. One thread registering twice without reverting.
        let (second_ok, second_err) = std::thread::spawn(|| {
            let (h1, _) = imp::raw_register();
            let (h2, err) = imp::raw_register();
            let ok = !h2.is_null();
            if ok {
                imp::raw_revert(h2);
            }
            if !h1.is_null() {
                imp::raw_revert(h1);
            }
            (ok, err)
        })
        .join()
        .unwrap();
        // 3. Concurrent registrations held by distinct threads.
        let n = 64;
        let hold = std::sync::Arc::new(std::sync::Barrier::new(n + 1));
        let (tx, rx) = std::sync::mpsc::channel();
        let threads: Vec<_> = (0..n)
            .map(|_| {
                let hold = std::sync::Arc::clone(&hold);
                let tx = tx.clone();
                std::thread::spawn(move || {
                    let (h, err) = imp::raw_register();
                    tx.send((!h.is_null(), err)).unwrap();
                    hold.wait();
                    if !h.is_null() {
                        imp::raw_revert(h);
                    }
                })
            })
            .collect();
        let held: Vec<_> = (0..n).map(|_| rx.recv().unwrap()).collect();
        hold.wait();
        for t in threads {
            t.join().unwrap();
        }
        let held_ok = held.iter().filter(|r| r.0).count();
        let held_err = held.iter().find(|r| !r.0).map_or(0, |r| r.1);
        eprintln!(
            "exit without revert: {failed_exit}/5000 failed (first error {first_err_exit}); \
             second registration on one thread: ok={second_ok} (error {second_err}); \
             concurrent: {held_ok}/{n} held (error {held_err})"
        );
    }

    // Diagnostic (run with --ignored --nocapture): what a callback thread's
    // one-time promotion costs, i.e. what the first render() on a cpal
    // callback thread pays (and, since the timer starts first, reports).
    #[test]
    #[ignore]
    fn promotion_cost_probe() {
        let mut us: Vec<f64> = (0..200)
            .map(|_| {
                std::thread::spawn(|| {
                    let t = std::time::Instant::now();
                    super::ensure_callback_thread_promoted();
                    let first = t.elapsed().as_secs_f64() * 1e6;
                    let t = std::time::Instant::now();
                    super::ensure_callback_thread_promoted();
                    (first, t.elapsed().as_secs_f64() * 1e6)
                })
                .join()
                .unwrap()
            })
            .map(|(first, again)| {
                assert!(again < first);
                first
            })
            .collect();
        us.sort_by(f64::total_cmp);
        eprintln!(
            "first-callback promotion: min {:.1} us, median {:.1} us, p95 {:.1} us, max {:.1} us \n             (128 frames at 48 kHz = 2667 us)",
            us[0],
            us[us.len() / 2],
            us[us.len() * 95 / 100],
            us[us.len() - 1]
        );
    }
}
