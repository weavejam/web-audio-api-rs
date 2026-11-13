//! Realtime-scheduling stress probe: renders a native-node graph on an
//! `AudioContext` with the `none` sink (a timer thread that calls the render
//! callback every 128 frames, like a device would) while `--busy` threads at
//! normal priority keep every CPU of the process busy, and reports how late each
//! quantum finished.
//!
//! `cargo run --release --example rt_load -- --workers 4 --busy 8 --seconds 10`
//!
//! A probe processor that every chain feeds into runs last in each quantum; it
//! records `now - frame / sample_rate`. The smallest such offset is the
//! baseline; a quantum is a *deadline miss* when it finished more than one
//! quantum (2.67 ms at 48 kHz) after the baseline, i.e. a device with a
//! one-quantum buffer would have underrun.

use std::hint::black_box;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use web_audio_api::context::{
    AudioContext, AudioContextLatencyCategory, AudioContextOptions, BaseAudioContext,
};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, BiquadFilterType, OscillatorType};
use web_audio_api::worklet::{
    AudioParamValues, AudioWorkletGlobalScope, AudioWorkletNode, AudioWorkletNodeOptions,
    AudioWorkletProcessor,
};

const SAMPLE_RATE: f32 = 48_000.;

type Offsets = Arc<Mutex<Vec<f64>>>;

struct Probe {
    offsets: Offsets,
    epoch: Instant,
}

impl AudioWorkletProcessor for Probe {
    type ProcessorOptions = Offsets;

    fn constructor(offsets: Offsets) -> Self {
        Self {
            offsets,
            epoch: Instant::now(),
        }
    }

    fn process<'a, 'b>(
        &mut self,
        inputs: &'b [&'a [&'a [f32]]],
        outputs: &'b mut [&'a mut [&'a mut [f32]]],
        _params: AudioParamValues<'b>,
        scope: &'b AudioWorkletGlobalScope,
    ) -> bool {
        let now = self.epoch.elapsed().as_secs_f64();
        let offset = now - scope.current_frame as f64 / scope.sample_rate as f64;
        if let Ok(mut offsets) = self.offsets.try_lock() {
            if offsets.len() < offsets.capacity() {
                offsets.push(offset);
            }
        }
        for (ic, oc) in inputs[0].iter().zip(outputs[0].iter_mut()) {
            oc.copy_from_slice(ic);
        }
        true
    }
}

fn arg(name: &str, default: usize) -> usize {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == &format!("--{name}"))
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

#[cfg(windows)]
#[link(name = "winmm")]
extern "system" {
    fn timeBeginPeriod(period: u32) -> u32;
}

fn main() {
    // 1 ms timer resolution: the none backend sleeps until each deadline.
    #[cfg(windows)]
    // SAFETY: no preconditions; the process exits without timeEndPeriod.
    unsafe {
        timeBeginPeriod(1);
    }
    let workers = arg("workers", 4);
    let busy = arg("busy", 8);
    let seconds = arg("seconds", 10);
    let chains = arg("chains", 48);
    let label = std::env::args()
        .position(|a| a == "--label")
        .and_then(|i| std::env::args().nth(i + 1))
        .unwrap_or_default();

    let context = AudioContext::new(AudioContextOptions {
        latency_hint: AudioContextLatencyCategory::Interactive,
        sample_rate: Some(SAMPLE_RATE),
        sink_id: "none".into(),
        render_parallelism: Some(workers),
        ..AudioContextOptions::default()
    });

    let quanta = seconds * SAMPLE_RATE as usize / 128 + 1024;
    let offsets: Offsets = Arc::new(Mutex::new(Vec::with_capacity(quanta)));
    let options = AudioWorkletNodeOptions {
        processor_options: Arc::clone(&offsets),
        ..AudioWorkletNodeOptions::default()
    };
    let probe = AudioWorkletNode::new::<Probe>(&context, options);
    probe.connect(&context.destination());

    // Independent chains, so the parallel executor spreads them over workers.
    for i in 0..chains {
        let mut osc = context.create_oscillator();
        osc.set_type(OscillatorType::Sawtooth);
        osc.frequency().set_value(55. + 7. * i as f32);
        let biquads: Vec<_> = (0..4)
            .map(|j| {
                let mut biquad = context.create_biquad_filter();
                biquad.set_type(BiquadFilterType::Lowpass);
                biquad.frequency().set_value(400. + 300. * j as f32);
                // a-rate automation keeps the per-sample coefficient path busy
                biquad
                    .frequency()
                    .linear_ramp_to_value_at_time(4000., seconds as f64 + 5.);
                biquad
            })
            .collect();
        osc.connect(&biquads[0]);
        for pair in biquads.windows(2) {
            pair[0].connect(&pair[1]);
        }
        let gain = context.create_gain();
        gain.gain().set_value(0.01);
        biquads[3].connect(&gain);
        gain.connect(&probe);
        osc.start();
    }

    // Let the graph settle before loading the machine.
    std::thread::sleep(Duration::from_millis(500));
    offsets.lock().unwrap().clear();
    let stats_before = context.playback_stats();
    let (u0, d0) = (
        stats_before.underrun_events(),
        stats_before.underrun_duration(),
    );

    let stop = Arc::new(AtomicBool::new(false));
    let spinners: Vec<_> = (0..busy)
        .map(|_| {
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || {
                let mut x = 1.0001_f64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..1000 {
                        x = black_box(x * 1.000_000_1 + 1e-9);
                    }
                }
                black_box(x);
            })
        })
        .collect();

    let started = Instant::now();
    std::thread::sleep(Duration::from_secs(seconds as u64));
    let wall = started.elapsed().as_secs_f64();
    let offs = offsets.lock().unwrap().clone();
    let stats = context.playback_stats();
    stop.store(true, Ordering::Relaxed);
    for s in spinners {
        let _ = s.join();
    }
    context.close_sync();

    let interval = 128. / SAMPLE_RATE as f64;
    let base = offs.iter().cloned().fold(f64::INFINITY, f64::min);
    let mut late: Vec<f64> = offs.iter().map(|o| (o - base) * 1e3).collect();
    late.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| late[((late.len() - 1) as f64 * p) as usize];
    let misses = late.iter().filter(|&&l| l > interval * 1e3).count();
    let misses4 = late.iter().filter(|&&l| l > 4. * interval * 1e3).count();
    let expected = (wall / interval) as usize;
    println!(
        "{{\"label\":\"{label}\",\"workers\":{workers},\"busy\":{busy},\"chains\":{chains},\"seconds\":{wall:.2},\
\"quanta\":{},\"expectedQuanta\":{expected},\"misses\":{misses},\"misses4\":{misses4},\
\"p50Ms\":{:.3},\"p99Ms\":{:.3},\"p999Ms\":{:.3},\"maxMs\":{:.3},\
\"underrunEvents\":{},\"underrunMs\":{:.3}}}",
        late.len(),
        pct(0.5),
        pct(0.99),
        pct(0.999),
        late.last().copied().unwrap_or(0.),
        stats.underrun_events() - u0,
        (stats.underrun_duration() - d0) * 1e3,
    );
}
