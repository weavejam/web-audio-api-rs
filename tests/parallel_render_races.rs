//! Regression tests for the multicore (`WEB_AUDIO_RS_PARALLEL`) render executor
//! on graph shapes the hand-built `graph.rs` tests do not reach:
//!
//! * nodes that keep render buffers inside their processor state (the delay
//!   line, the compressor's look-ahead ring buffer) while the plan is
//!   recomputed: every freed node triggers a re-plan whose round-robin can move
//!   a surviving node to another worker, leaving those buffers bound to the
//!   previous worker's allocator. This crashed the render thread
//!   (`RefCell already borrowed` in the buffer pool, or an access violation
//!   from a racy `Rc` count).
//! * feedback loops through a `DelayNode`: cycle breaking clears the delay
//!   writer's edge to its reader for good, so the two halves (which share the
//!   delay line) were planned as unrelated partitions. Once the feedback path is
//!   gone they run concurrently on different workers.
//!
//! The threaded output must also stay byte-identical to the serial render.
//!
//! The executor is only selectable for an `OfflineAudioContext` through the
//! environment variable read at graph construction, so these tests live in
//! their own test binary and serialize every env access through `ENV_LOCK`.

use std::sync::Mutex;

use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, OscillatorType};

static ENV_LOCK: Mutex<()> = Mutex::new(());

const SAMPLE_RATE: f32 = 48_000.;

/// Create an offline context whose graph renders on `workers` threads
/// (`None` = the serial executor).
fn context(workers: Option<usize>, length: usize) -> OfflineAudioContext {
    let _guard = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    match workers {
        Some(n) => std::env::set_var("WEB_AUDIO_RS_PARALLEL", n.to_string()),
        None => std::env::remove_var("WEB_AUDIO_RS_PARALLEL"),
    }
    let ctx = OfflineAudioContext::new(2, length, SAMPLE_RATE);
    std::env::remove_var("WEB_AUDIO_RS_PARALLEL");
    ctx
}

/// Feedback-delay loops (each with a compressor in the loop, so both halves of
/// the loop hold render buffers in processor state) plus a swarm of short-lived
/// oscillators whose staggered ends free nodes - and hence re-plan the
/// partitions - all through the render.
fn build_feedback_graph(ctx: &OfflineAudioContext) {
    let dest = ctx.destination();
    for i in 0..12 {
        let mut osc = ctx.create_oscillator();
        osc.set_type(OscillatorType::Sawtooth);
        osc.frequency().set_value(55. * (i + 1) as f32);

        let delay = ctx.create_delay(1.);
        delay.delay_time().set_value(0.031 + 0.017 * i as f32);

        let comp = ctx.create_dynamics_compressor();
        let feedback = ctx.create_gain();
        feedback.gain().set_value(0.55);
        let out = ctx.create_gain();
        out.gain().set_value(0.08);

        osc.connect(&delay);
        delay.connect(&comp);
        comp.connect(&feedback);
        feedback.connect(&delay); // cycle: broken at the delay
        delay.connect(&out);
        out.connect(&dest);

        // The excitation stops early so the loop keeps ringing on its feedback
        // alone while the oscillator node itself is freed (-> re-plan).
        osc.start();
        osc.stop_at(0.15 + 0.05 * i as f64);
    }

    let bus = ctx.create_gain();
    bus.gain().set_value(0.02);
    bus.connect(&dest);
    for i in 0..48 {
        let mut osc = ctx.create_oscillator();
        osc.frequency().set_value(200. + 13. * i as f32);
        osc.connect(&bus);
        osc.start_at(0.01 * i as f64);
        osc.stop_at(0.05 + 0.021 * i as f64);
    }
}

fn render_bits(workers: Option<usize>, length: usize) -> Vec<Vec<u32>> {
    let mut ctx = context(workers, length);
    build_feedback_graph(&ctx);
    let buffer = ctx.start_rendering_sync();
    (0..buffer.number_of_channels())
        .map(|c| {
            buffer
                .get_channel_data(c)
                .iter()
                .map(|s| s.to_bits())
                .collect()
        })
        .collect()
}

/// Feedback-delay loops whose feedback is disconnected mid-render. The cycle
/// breaker cleared each delay writer's edge to its reader for good, so once the
/// loop is gone nothing in the graph links (or orders) the two halves any more:
/// both sit at the first DAG level and, unless the planner keeps them together,
/// run concurrently on different workers against their shared delay line.
fn build_detached_feedback_graph(ctx: &mut OfflineAudioContext) {
    let dest = ctx.destination();
    let mut feedbacks = Vec::new();
    for i in 0..12 {
        let mut osc = ctx.create_oscillator();
        osc.set_type(OscillatorType::Square);
        osc.frequency().set_value(70. * (i + 1) as f32);

        let delay = ctx.create_delay(1.);
        delay.delay_time().set_value(0.023 + 0.011 * i as f32);

        let comp = ctx.create_dynamics_compressor();
        let feedback = ctx.create_gain();
        feedback.gain().set_value(0.5);
        let out = ctx.create_gain();
        out.gain().set_value(0.08);

        osc.connect(&delay);
        delay.connect(&comp);
        comp.connect(&feedback);
        feedback.connect(&delay);
        delay.connect(&out);
        out.connect(&dest);
        osc.start();

        feedbacks.push(feedback);
    }
    ctx.suspend_sync(0.4, move |_| {
        for feedback in &feedbacks {
            feedback.disconnect();
        }
    });
}

fn render_detached_bits(workers: Option<usize>, length: usize) -> Vec<Vec<u32>> {
    let mut ctx = context(workers, length);
    build_detached_feedback_graph(&mut ctx);
    let buffer = ctx.start_rendering_sync();
    (0..buffer.number_of_channels())
        .map(|c| {
            buffer
                .get_channel_data(c)
                .iter()
                .map(|s| s.to_bits())
                .collect()
        })
        .collect()
}

fn assert_identical(threaded: &[Vec<u32>], serial: &[Vec<u32>], what: &str) {
    for (c, (got, want)) in threaded.iter().zip(serial).enumerate() {
        if got != want {
            let first = got.iter().zip(want).position(|(a, b)| a != b).unwrap();
            panic!(
                "{what} channel={c}: threaded render diverges from serial at frame {first} \
                 (quantum {})",
                first / 128
            );
        }
    }
}

#[test]
fn detached_feedback_delays_render_byte_identical_across_workers() {
    let length = SAMPLE_RATE as usize; // 1 s, the detach happens at 0.4 s
    let serial = render_detached_bits(None, length);
    for workers in [2usize, 4, 8] {
        for run in 0..3 {
            let threaded = render_detached_bits(Some(workers), length);
            assert_identical(&threaded, &serial, &format!("workers={workers} run={run}"));
        }
    }
}

#[test]
fn feedback_delay_loops_render_byte_identical_across_workers() {
    // 2 s: long enough for every scheduled stop (and its re-plan) to happen
    // while the feedback loops are still ringing.
    let length = (SAMPLE_RATE as usize) * 2;
    let serial = render_bits(None, length);
    assert!(
        serial[0].iter().any(|&b| f32::from_bits(b) != 0.),
        "reference render must not be silent"
    );

    for workers in [2usize, 4, 8] {
        // Several runs: the original failure was a timing-dependent race.
        for run in 0..3 {
            let threaded = render_bits(Some(workers), length);
            assert_identical(&threaded, &serial, &format!("workers={workers} run={run}"));
        }
    }
}

/// Every render worker flushes denormals, as worker 0 does (`tests/denormals.rs`):
/// FTZ/DAZ are per-thread flags that spawned workers do not inherit, so a chain
/// that a spawned worker renders used to keep its subnormal (and differ from the
/// serial render).
#[test]
fn parallel_workers_flush_denormals_like_serial() {
    let render = |workers: Option<usize>| {
        let mut ctx = context(workers, 128 * 8);
        let dest = ctx.destination();
        // Independent chains, so the planner spreads them over every worker.
        for _ in 0..16 {
            let mut signal = ctx.create_constant_source();
            signal.start();
            let gain1 = ctx.create_gain();
            gain1.gain().set_value(0.001);
            let gain2 = ctx.create_gain();
            gain2.gain().set_value(f32::MIN_POSITIVE); // 0.001 * MIN_POSITIVE is subnormal
            let gain3 = ctx.create_gain();
            gain3.gain().set_value(f32::MAX);
            signal.connect(&gain1);
            gain1.connect(&gain2);
            gain2.connect(&gain3);
            gain3.connect(&dest);
        }
        let out = ctx.start_rendering_sync();
        (0..2)
            .flat_map(|c| out.get_channel_data(c).to_vec())
            .collect::<Vec<f32>>()
    };
    let serial = render(None);
    assert!(serial.iter().all(|&s| s == 0.), "serial flushes denormals");
    let parallel = render(Some(4));
    let nonzero = parallel.iter().filter(|&&s| s != 0.).count();
    assert_eq!(
        nonzero, 0,
        "parallel render kept {nonzero} denormal-derived samples"
    );
}

/// Windows: an offline render on the worker pool runs its calling thread
/// (worker 0, which the realtime workers wait on) at realtime priority for the
/// duration of the render and restores it afterwards; a serial render leaves
/// the caller's priority alone.
#[cfg(windows)]
#[test]
fn offline_pool_render_promotes_the_calling_thread_for_the_render() {
    use std::sync::atomic::{AtomicI32, Ordering};
    use std::sync::Arc;
    use web_audio_api::worklet::{
        AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
        AudioWorkletProcessor, ThreadAffinity,
    };

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn GetThreadPriority(thread: *mut std::ffi::c_void) -> i32;
    }
    fn priority() -> i32 {
        // SAFETY: pseudo-handle valid for the calling thread.
        unsafe { GetThreadPriority(GetCurrentThread()) }
    }

    /// Records the priority of the thread it runs on (pinned to worker 0).
    struct Probe(Arc<AtomicI32>);
    impl AudioWorkletProcessor for Probe {
        type ProcessorOptions = Arc<AtomicI32>;
        fn constructor(seen: Arc<AtomicI32>) -> Self {
            Self(seen)
        }
        fn thread_affinity() -> ThreadAffinity {
            ThreadAffinity::RenderThread
        }
        fn process<'a, 'b>(
            &mut self,
            _inputs: &'b [&'a [&'a [f32]]],
            _outputs: &'b mut [&'a mut [&'a mut [f32]]],
            _params: AudioParamValues<'b>,
            _scope: &'b AudioWorkletGlobalScope,
        ) -> bool {
            self.0.store(priority(), Ordering::Relaxed);
            true
        }
    }

    let render = |workers: Option<usize>| {
        std::thread::spawn(move || {
            let mut ctx = context(workers, 128 * 4);
            let seen = Arc::new(AtomicI32::new(i32::MIN));
            let options = AudioWorkletNodeOptions {
                processor_options: Arc::clone(&seen),
                ..AudioWorkletNodeOptions::default()
            };
            let probe = AudioWorkletNode::new::<Probe>(&ctx, options);
            probe.connect(&ctx.destination());
            for _ in 0..8 {
                let mut osc = ctx.create_oscillator();
                osc.connect(&ctx.destination());
                osc.start();
            }
            let before = priority();
            let _ = ctx.start_rendering_sync();
            (before, seen.load(Ordering::Relaxed), priority())
        })
        .join()
        .unwrap()
    };
    const TIME_CRITICAL: i32 = 15;

    let (before, during, after) = render(Some(4));
    assert_eq!(before, 0);
    assert_eq!(
        during, TIME_CRITICAL,
        "worker 0 is promoted while rendering"
    );
    assert_eq!(after, before, "the caller's priority is restored");

    let (before, during, after) = render(None);
    assert_eq!(
        (before, during, after),
        (0, 0, 0),
        "serial render: unchanged"
    );
}
