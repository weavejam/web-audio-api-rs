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
