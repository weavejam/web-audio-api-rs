//! Render-loop cost of AudioParams: offline renders of track-like chains whose
//! nodes are mostly unautomated params (as on a DAW project), timed in-process.
//!
//! `cargo run --release --example bench_params -- [tracks] [seconds] [rounds]`
//!
//! Each round renders the same graph with `WEB_AUDIO_RS_INLINE_PARAMS` = 0 and
//! = 1 (interleaved) and prints the wall time of each; the summary is the
//! fastest round per side. Outputs are compared bit for bit.
use std::time::Instant;

use web_audio_api::context::{BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, BiquadFilterType};

fn render(tracks: usize, seconds: f32) -> (f64, f64, Vec<f32>) {
    let sr = 48000.;
    let mut ctx = OfflineAudioContext::new(2, (seconds * sr) as usize, sr);
    let master = ctx.create_gain();
    master.connect(&ctx.destination());
    for t in 0..tracks {
        // source -> 4-band EQ -> 4 gains -> panner -> track gain -> master
        let mut osc = ctx.create_oscillator();
        osc.frequency().set_value(55. * (1 + t % 7) as f32);
        osc.start();
        let mut prev: Box<dyn AudioNode> = Box::new(osc);
        for b in 0..4 {
            let mut f = ctx.create_biquad_filter();
            f.set_type(BiquadFilterType::Peaking);
            f.frequency().set_value(200. * (b + 1) as f32);
            f.gain().set_value(1.5);
            prev.connect(&f);
            prev = Box::new(f);
        }
        for g in 0..4 {
            let gain = ctx.create_gain();
            gain.gain().set_value(0.9);
            if t % 8 == 0 && g == 0 {
                // a few automated params, as in a real project
                gain.gain().linear_ramp_to_value_at_time(0.2, seconds as f64);
            }
            prev.connect(&gain);
            prev = Box::new(gain);
        }
        let pan = ctx.create_stereo_panner();
        pan.pan().set_value(((t % 5) as f32 - 2.) / 3.);
        prev.connect(&pan);
        let track = ctx.create_gain();
        track.gain().set_value(1. / tracks as f32);
        pan.connect(&track);
        track.connect(&master);
    }
    let start = Instant::now();
    let cpu = thread_cpu_ms();
    let out = ctx.start_rendering_sync();
    let cpu = thread_cpu_ms() - cpu;
    let ms = start.elapsed().as_secs_f64() * 1000.;
    (ms, cpu, out.get_channel_data(0).to_vec())
}

fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().unwrap())
        .collect();
    let tracks = args.first().copied().unwrap_or(32);
    let seconds = args.get(1).copied().unwrap_or(10) as f32;
    let rounds = args.get(2).copied().unwrap_or(7);

    let mut best = [f64::MAX; 2];
    let mut worst = [0f64; 2];
    let mut best_cpu = [f64::MAX; 2];
    let mut reference: Option<Vec<u32>> = None;
    for round in 0..rounds {
        for (side, flag) in ["0", "1"].iter().enumerate() {
            std::env::set_var("WEB_AUDIO_RS_INLINE_PARAMS", flag);
            let (ms, cpu, out) = render(tracks, seconds);
            let bits: Vec<u32> = out.iter().map(|s| s.to_bits()).collect();
            match &reference {
                None => reference = Some(bits),
                Some(r) => assert!(*r == bits, "output differs (inline={flag})"),
            }
            println!("round={round} inline={flag} ms={ms:.1} cpu_ms={cpu:.1}");
            best_cpu[side] = best_cpu[side].min(cpu);
            best[side] = best[side].min(ms);
            worst[side] = worst[side].max(ms);
        }
    }
    println!(
        "tracks={tracks} seconds={seconds} rounds={rounds} min_ms off={:.1} on={:.1} change={:+.1}% (max off={:.1} on={:.1}) min_cpu_ms off={:.1} on={:.1} change={:+.1}%",
        best[0],
        best[1],
        (best[1] / best[0] - 1.) * 100.,
        worst[0],
        worst[1],
        best_cpu[0],
        best_cpu[1],
        (best_cpu[1] / best_cpu[0] - 1.) * 100.
    );
}

/// CPU time of the calling thread in ms (Windows), so a loaded machine's
/// preemption does not count; wall time elsewhere.
#[cfg(windows)]
fn thread_cpu_ms() -> f64 {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime {
        lo: u32,
        hi: u32,
    }
    extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn GetThreadTimes(
            h: *mut std::ffi::c_void,
            c: *mut FileTime,
            e: *mut FileTime,
            k: *mut FileTime,
            u: *mut FileTime,
        ) -> i32;
    }
    let (mut c, mut e, mut k, mut u) = Default::default();
    unsafe { GetThreadTimes(GetCurrentThread(), &mut c, &mut e, &mut k, &mut u) };
    let t = |f: &FileTime| ((f.hi as u64) << 32 | f.lo as u64) as f64 / 10_000.;
    t(&k) + t(&u)
}
#[cfg(not(windows))]
fn thread_cpu_ms() -> f64 {
    0.
}
