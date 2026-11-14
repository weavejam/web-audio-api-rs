use std::any::Any;

use fft_convolver::FFTConvolver;

use crate::buffer::AudioBuffer;
use crate::context::{AudioContextRegistration, BaseAudioContext};
use crate::render::{
    AudioParamValues, AudioProcessor, AudioRenderQuantum, AudioWorkletGlobalScope,
};
use crate::RENDER_QUANTUM_SIZE;

use super::{AudioNode, AudioNodeOptions, ChannelConfig, ChannelCountMode, ChannelInterpretation};

enum Convolver {
    Fft(Box<FFTConvolver<f32>>),
    Sparse(SparseConvolver),
}

impl Convolver {
    fn new(partition_size: usize, impulse: &[f32]) -> Self {
        const MAX_DIRECT_TAPS: usize = 8;
        let taps: Vec<_> = impulse
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, gain)| *gain != 0.)
            .take(MAX_DIRECT_TAPS + 1)
            .collect();
        if taps.len() <= MAX_DIRECT_TAPS && taps.iter().all(|(_, gain)| gain.is_finite()) {
            // Direct FIR differs from FFT only by relative FFT roundoff: the
            // absolute error scales with input level and impulse-response gain.
            // Allocate on the control thread, never in process(). A 1M-frame
            // single-tap IR still needs a next_power_of_two(len) ring (4 MiB).
            let history_length = impulse.len().max(1).next_power_of_two();
            return Self::Sparse(SparseConvolver {
                taps,
                history: vec![0.; history_length],
                position: 0,
            });
        }

        let mut fft = FFTConvolver::default();
        fft.init(partition_size, impulse)
            .expect("Unable to initialize convolution engine");
        Self::Fft(Box::new(fft))
    }

    fn process(
        &mut self,
        input: &[f32],
        output: &mut [f32],
    ) -> Result<(), fft_convolver::FFTConvolverError> {
        match self {
            Self::Fft(fft) => fft.process(input, output),
            Self::Sparse(sparse) => {
                sparse.process(input, output);
                Ok(())
            }
        }
    }
}

struct SparseConvolver {
    taps: Vec<(usize, f32)>,
    history: Vec<f32>,
    position: usize,
}

impl SparseConvolver {
    fn process(&mut self, input: &[f32], output: &mut [f32]) {
        let mask = self.history.len() - 1;
        for (&sample, out) in input.iter().zip(output) {
            self.history[self.position] = sample;
            let mut sum = 0.;
            for &(delay, gain) in &self.taps {
                sum += self.history[self.position.wrapping_sub(delay) & mask] * gain;
            }
            *out = sum;
            self.position = (self.position + 1) & mask;
        }
    }
}

/// Scale buffer by an equal-power normalization
// see - <https://webaudio.github.io/web-audio-api/#dom-convolvernode-normalize>
fn normalize_buffer(buffer: &AudioBuffer) -> f32 {
    let gain_calibration = 0.00125;
    let gain_calibration_sample_rate = 44100.;
    let min_power = 0.000125;

    // Normalize by RMS power.
    let number_of_channels = buffer.number_of_channels();
    let length = buffer.length();
    let sample_rate = buffer.sample_rate();

    let mut power: f32 = buffer
        .channels()
        .iter()
        .map(|c| c.as_slice().iter().map(|&s| s * s).sum::<f32>())
        .sum();

    power = (power / (number_of_channels * length) as f32).sqrt();

    // Protect against accidental overload.
    if !power.is_finite() || power.is_nan() || power < min_power {
        power = min_power;
    }

    let mut scale = 1. / power;

    // Calibrate to make perceived volume same as unprocessed.
    scale *= gain_calibration;

    // Scale depends on sample-rate.
    scale *= gain_calibration_sample_rate / sample_rate;

    // True-stereo compensation.
    if number_of_channels == 4 {
        scale *= 0.5;
    }

    scale
}

/// `ConvolverNode` options
//dictionary ConvolverOptions : AudioNodeOptions {
//  AudioBuffer? buffer;
//  boolean disableNormalization = false;
//};
#[derive(Clone, Debug)]
pub struct ConvolverOptions {
    /// The desired buffer for the ConvolverNode
    pub buffer: Option<AudioBuffer>,
    /// The opposite of the desired initial value for the normalize attribute
    pub disable_normalization: bool,
    /// AudioNode options
    pub audio_node_options: AudioNodeOptions,
}

impl Default for ConvolverOptions {
    fn default() -> Self {
        Self {
            buffer: None,
            disable_normalization: false,
            audio_node_options: AudioNodeOptions {
                channel_count: 2,
                channel_count_mode: ChannelCountMode::ClampedMax,
                channel_interpretation: ChannelInterpretation::Speakers,
            },
        }
    }
}

/// Assert that the channel count is valid for the ConvolverNode
/// see <https://webaudio.github.io/web-audio-api/#audionode-channelcount-constraints>
///
/// # Panics
///
/// This function panics if given count is greater than 2
///
#[track_caller]
#[inline(always)]
fn assert_valid_channel_count(count: usize) {
    assert!(
        count <= 2,
        "NotSupportedError - ConvolverNode channel count cannot be greater than two"
    );
}

/// Assert that the channel count mode is valid for the ConvolverNode
/// see <https://webaudio.github.io/web-audio-api/#audionode-channelcountmode-constraints>
///
/// # Panics
///
/// This function panics if given count mode is [`ChannelCountMode::Max`]
///
#[track_caller]
#[inline(always)]
fn assert_valid_channel_count_mode(mode: ChannelCountMode) {
    assert_ne!(
        mode,
        ChannelCountMode::Max,
        "NotSupportedError - ConvolverNode channel count mode cannot be set to max"
    );
}

/// Processing node which applies a linear convolution effect given an impulse response.
///
/// - MDN documentation: <https://developer.mozilla.org/en-US/docs/Web/API/ConvolverNode>
/// - specification: <https://webaudio.github.io/web-audio-api/#ConvolverNode>
/// - see also: [`BaseAudioContext::create_convolver`]
///
/// The current implementation only handles mono-to-mono convolutions. The provided impulse
/// response buffer and the input signal will be downmixed appropriately.
///
/// # Usage
///
/// ```no_run
/// use std::fs::File;
///
/// use web_audio_api::context::{AudioContext, BaseAudioContext};
/// use web_audio_api::node::{AudioNode, AudioScheduledSourceNode, ConvolverNode, ConvolverOptions};
///
/// let context = AudioContext::default();
/// let file = File::open("samples/vocals-dry.wav").unwrap();
/// let audio_buffer = context.decode_audio_data_sync(file).unwrap();
///
/// let impulse_file = File::open("samples/small-room-response.wav").unwrap();
/// let impulse_buffer = context.decode_audio_data_sync(impulse_file).unwrap();
///
/// let mut src = context.create_buffer_source();
/// src.set_buffer(audio_buffer);
///
/// let mut convolve = ConvolverNode::new(&context, ConvolverOptions::default());
/// convolve.set_buffer(impulse_buffer);
///
/// src.connect(&convolve);
/// convolve.connect(&context.destination());
/// src.start();
/// std::thread::sleep(std::time::Duration::from_millis(4_000));
/// ```
///
/// # Examples
///
/// - `cargo run --release --example convolution`
///
#[derive(Debug)]
pub struct ConvolverNode {
    /// Represents the node instance and its associated audio context
    registration: AudioContextRegistration,
    /// Info about audio node channel configuration
    channel_config: ChannelConfig,
    /// Perform equal power normalization on response buffer
    normalize: bool,
    /// The response buffer, nullable
    buffer: Option<AudioBuffer>,
}

impl AudioNode for ConvolverNode {
    fn registration(&self) -> &AudioContextRegistration {
        &self.registration
    }

    fn channel_config(&self) -> &ChannelConfig {
        &self.channel_config
    }

    fn number_of_inputs(&self) -> usize {
        1
    }

    fn number_of_outputs(&self) -> usize {
        1
    }

    // see <https://webaudio.github.io/web-audio-api/#audionode-channelcount-constraints>
    fn set_channel_count(&self, count: usize) {
        assert_valid_channel_count(count);
        self.channel_config.set_count(count, self.registration());
    }

    // see <https://webaudio.github.io/web-audio-api/#audionode-channelcountmode-constraints>
    fn set_channel_count_mode(&self, mode: ChannelCountMode) {
        assert_valid_channel_count_mode(mode);
        self.channel_config
            .set_count_mode(mode, self.registration());
    }
}

impl ConvolverNode {
    /// returns a `ConvolverNode` instance
    ///
    /// # Arguments
    ///
    /// * `context` - audio context in which the audio node will live.
    /// * `options` - convolver options
    ///
    /// # Panics
    ///
    /// Panics when an AudioBuffer is provided via the `ConvolverOptions` with a sample rate
    /// different from the audio context sample rate.
    pub fn new<C: BaseAudioContext>(context: &C, options: ConvolverOptions) -> Self {
        let ConvolverOptions {
            buffer,
            disable_normalization,
            audio_node_options,
        } = options;

        assert_valid_channel_count(audio_node_options.channel_count);
        assert_valid_channel_count_mode(audio_node_options.channel_count_mode);

        let mut node = context.base().register(move |registration| {
            let renderer = ConvolverRenderer {
                convolvers: None,
                impulse_length: 0,
                impulse_number_of_channels: 0,
                tail_count: 0,
            };

            let node = Self {
                registration,
                channel_config: audio_node_options.into(),
                normalize: !disable_normalization,
                buffer: None,
            };

            (node, Box::new(renderer))
        });

        // renderer has been sent to render thread, we can send it messages
        if let Some(buffer) = buffer {
            node.set_buffer(buffer);
        }

        node
    }

    /// Get the current impulse response buffer
    pub fn buffer(&self) -> Option<&AudioBuffer> {
        self.buffer.as_ref()
    }

    /// Set or update the impulse response buffer
    ///
    /// # Panics
    ///
    /// Panics when the sample rate of the provided AudioBuffer differs from the audio context
    /// sample rate.
    pub fn set_buffer(&mut self, buffer: AudioBuffer) {
        // If the buffer number of channels is not 1, 2, 4, or if the sample-rate of the buffer is
        // not the same as the sample-rate of its associated BaseAudioContext, a NotSupportedError
        // MUST be thrown.

        let sample_rate = buffer.sample_rate();
        assert_eq!(
            sample_rate,
            self.context().sample_rate(),
            "NotSupportedError - sample rate of the convolution buffer must match the audio context"
        );

        let number_of_channels = buffer.number_of_channels();
        assert!(
            [1, 2, 4].contains(&number_of_channels),
            "NotSupportedError - the convolution buffer must consist of 1, 2 or 4 channels"
        );

        // normalize before padding because the length of the buffer affects the scale
        let scale = if self.normalize {
            normalize_buffer(&buffer)
        } else {
            1.
        };

        let mut convolvers = Vec::<Convolver>::new();
        // @note - value defined by "rule of thumb", to be explored further
        let partition_size = RENDER_QUANTUM_SIZE * 8;

        // Handle multichannel IR
        // cf. https://webaudio.github.io/web-audio-api/#Convolution-channel-configurations
        // Note that in case of mono IR we create 2 convolvers to properly handle stereo input
        for index in 0..number_of_channels.max(2) {
            // make sure we don't try to access an inexisting channel, cf. note above
            let channel = index.min(number_of_channels - 1);

            let mut scaled_channel = vec![0.; buffer.length()];
            scaled_channel
                .iter_mut()
                .zip(buffer.get_channel_data(channel))
                .for_each(|(o, i)| *o = *i * scale);

            convolvers.push(Convolver::new(partition_size, &scaled_channel));
        }

        let msg = ConvolverInfosMessage {
            convolvers: Some(convolvers),
            impulse_length: buffer.length(),
            impulse_number_of_channels: number_of_channels,
        };

        self.registration.post_message(msg);
        self.buffer = Some(buffer);
    }

    /// Denotes if the response buffer will be scaled with an equal-power normalization
    pub fn normalize(&self) -> bool {
        self.normalize
    }

    /// Update the `normalize` setting. This will only have an effect when `set_buffer` is called.
    pub fn set_normalize(&mut self, value: bool) {
        self.normalize = value;
    }
}

struct ConvolverInfosMessage {
    convolvers: Option<Vec<Convolver>>,
    impulse_length: usize,
    impulse_number_of_channels: usize,
}

struct ConvolverRenderer {
    convolvers: Option<Vec<Convolver>>,
    impulse_length: usize,
    impulse_number_of_channels: usize,
    tail_count: usize,
}

impl AudioProcessor for ConvolverRenderer {
    fn process(
        &mut self,
        inputs: &[AudioRenderQuantum],
        outputs: &mut [AudioRenderQuantum],
        _params: AudioParamValues<'_>,
        _scope: &AudioWorkletGlobalScope,
    ) -> bool {
        #[cfg(feature = "param-profile")]
        let _timer = crate::param_profile::Timer::new(7);
        // single input/output node
        let input = &inputs[0];
        let output = &mut outputs[0];

        // handle tail time and active processing
        // return early if input is silent and we reached the end of the convolution
        if input.is_silent() {
            if self.tail_count >= self.impulse_length {
                output.make_silent();
                return false;
            }

            self.tail_count += RENDER_QUANTUM_SIZE;
        } else {
            self.tail_count = 0;
        }

        let convolvers = match &mut self.convolvers {
            None => {
                // no convolution buffer set, passthrough
                *output = input.clone();
                return !input.is_silent();
            }
            Some(convolvers) => convolvers,
        };

        // https://webaudio.github.io/web-audio-api/#Convolution-channel-configurations
        match (input.number_of_channels(), self.impulse_number_of_channels) {
            (1, 1) => {
                output.set_number_of_channels(1);

                let i = &input.channel_data(0)[..];
                let o = &mut output.channel_data_mut(0)[..];
                let _ = convolvers[0].process(i, o);
            }
            (1, 2) => {
                output.set_number_of_channels(2);

                let i = &input.channel_data(0)[..];

                let o_left = &mut output.channel_data_mut(0)[..];
                let _ = convolvers[0].process(i, o_left);

                let o_right = &mut output.channel_data_mut(1)[..];
                let _ = convolvers[1].process(i, o_right);
            }
            (2, 1) => {
                output.set_number_of_channels(2);

                let i_left = &input.channel_data(0)[..];
                let o_left = &mut output.channel_data_mut(0)[..];
                let _ = convolvers[0].process(i_left, o_left);

                let i_right = &input.channel_data(1)[..];
                let o_right = &mut output.channel_data_mut(1)[..];
                let _ = convolvers[1].process(i_right, o_right);
            }
            (2, 2) => {
                output.set_number_of_channels(2);

                let i_left = &input.channel_data(0)[..];
                let o_left = &mut output.channel_data_mut(0)[..];
                let _ = convolvers[0].process(i_left, o_left);

                let i_right = &input.channel_data(1)[..];
                let o_right = &mut output.channel_data_mut(1)[..];
                let _ = convolvers[1].process(i_right, o_right);
            }
            (2, 4) => {
                output.set_number_of_channels(4);

                let i_left = &input.channel_data(0)[..];

                let o_0 = &mut output.channel_data_mut(0)[..];
                let _ = convolvers[0].process(i_left, o_0);
                let o_1 = &mut output.channel_data_mut(1)[..];
                let _ = convolvers[1].process(i_left, o_1);

                let i_right = &input.channel_data(1)[..];

                let o_2 = &mut output.channel_data_mut(2)[..];
                let _ = convolvers[2].process(i_right, o_2);
                let o_3 = &mut output.channel_data_mut(3)[..];
                let _ = convolvers[3].process(i_right, o_3);

                // mix output back to stereo
                let o_2 = output.channel_data(2).clone();
                let o_3 = output.channel_data(3).clone();

                output
                    .channel_data_mut(0)
                    .iter_mut()
                    .zip(o_2.iter())
                    .for_each(|(l, sl)| *l += *sl);

                output
                    .channel_data_mut(1)
                    .iter_mut()
                    .zip(o_3.iter())
                    .for_each(|(r, sr)| *r += *sr);

                output.set_number_of_channels(2);
            }
            (1, 4) => {
                output.set_number_of_channels(4);

                let i = &input.channel_data(0)[..];

                let o_0 = &mut output.channel_data_mut(0)[..];
                let _ = convolvers[0].process(i, o_0);
                let o_1 = &mut output.channel_data_mut(1)[..];
                let _ = convolvers[1].process(i, o_1);
                let o_2 = &mut output.channel_data_mut(2)[..];
                let _ = convolvers[2].process(i, o_2);
                let o_3 = &mut output.channel_data_mut(3)[..];
                let _ = convolvers[3].process(i, o_3);

                // mix output back to stereo
                let o_2 = output.channel_data(2).clone();
                let o_3 = output.channel_data(3).clone();

                output
                    .channel_data_mut(0)
                    .iter_mut()
                    .zip(o_2.iter())
                    .for_each(|(l, sl)| *l += *sl);

                output
                    .channel_data_mut(1)
                    .iter_mut()
                    .zip(o_3.iter())
                    .for_each(|(r, sr)| *r += *sr);

                output.set_number_of_channels(2);
            }
            _ => unreachable!(),
        }

        true
    }

    fn onmessage(&mut self, msg: &mut dyn Any) {
        if let Some(msg) = msg.downcast_mut::<ConvolverInfosMessage>() {
            let ConvolverInfosMessage {
                convolvers,
                impulse_length,
                impulse_number_of_channels,
            } = msg;
            // Avoid deallocation in the render thread by swapping the convolver.
            std::mem::swap(&mut self.convolvers, convolvers);
            self.impulse_length = *impulse_length;
            self.impulse_number_of_channels = *impulse_number_of_channels;

            return;
        }

        log::warn!("ConvolverRenderer: Dropping incoming message {msg:?}");
    }
}

#[cfg(test)]
mod tests {
    use float_eq::assert_float_eq;

    use crate::context::{BaseAudioContext, OfflineAudioContext};
    use crate::node::{AudioBufferSourceNode, AudioBufferSourceOptions, AudioScheduledSourceNode};

    use super::*;

    #[test]
    fn test_sparse_zero_latency_and_tail_across_ring_wraps() {
        for delay in [0, 127, 128, 1024, 4095] {
            let mut impulse = vec![0.; delay + 3];
            impulse[delay..].copy_from_slice(&[0.625, 0.25, 0.125]);
            let mut convolver = Convolver::new(1024, &impulse);
            assert!(matches!(convolver, Convolver::Sparse(_)));
            let mut input = vec![0.; 3 * impulse.len() + 256];
            input[0] = 1.;
            input[127] = -0.5;
            input[impulse.len() + 2] = 0.25;
            let mut output = vec![0.; input.len()];
            for (input, output) in input.chunks(128).zip(output.chunks_mut(128)) {
                convolver.process(input, output).unwrap();
            }
            for (frame, &sample) in output.iter().enumerate() {
                let expected: f32 = impulse
                    .iter()
                    .enumerate()
                    .filter(|(tap, _)| *tap <= frame)
                    .map(|(tap, gain)| gain * input[frame - tap])
                    .sum();
                assert_eq!(sample, expected, "delay {delay}, frame {frame}");
            }
        }
    }

    #[test]
    fn test_sparse_and_fft_match_direct_reference_with_streaming_input() {
        let mut state = 42u32;
        let mut random = || {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state as f64 / u32::MAX as f64 * 2. - 1.) as f32
        };
        for tap_count in [0, 1, 3, 8, 9] {
            let mut impulse = vec![0.; 1027];
            for tap in 0..tap_count {
                impulse[tap * 128 + 2] = random() / 8.;
            }
            let mut convolver = Convolver::new(1024, &impulse);
            assert_eq!(matches!(convolver, Convolver::Sparse(_)), tap_count <= 8);
            let mut input: Vec<_> = (0..4096).map(|_| random()).collect();
            input.extend(vec![0.; 2048]);
            let mut actual = vec![0.; input.len()];
            let expected: Vec<f32> = (0..input.len())
                .map(|frame| {
                    impulse
                        .iter()
                        .enumerate()
                        .filter(|(tap, gain)| *tap <= frame && **gain != 0.)
                        .map(|(tap, gain)| input[frame - tap] * gain)
                        .sum()
                })
                .collect();
            for (input, actual) in input.chunks(128).zip(actual.chunks_mut(128)) {
                convolver.process(input, actual).unwrap();
            }
            assert_float_eq!(actual[..], expected[..], abs_all <= 1e-6);
        }
    }

    fn node_with_backend(
        context: &OfflineAudioContext,
        impulse: AudioBuffer,
        normalize: bool,
        force_fft: bool,
    ) -> ConvolverNode {
        let scale = if normalize {
            normalize_buffer(&impulse)
        } else {
            1.
        };
        let node = ConvolverNode::new(
            context,
            ConvolverOptions {
                buffer: Some(impulse.clone()),
                disable_normalization: !normalize,
                ..ConvolverOptions::default()
            },
        );
        if force_fft {
            let convolvers = (0..impulse.number_of_channels().max(2))
                .map(|channel| {
                    let scaled: Vec<_> = impulse
                        .get_channel_data(channel.min(impulse.number_of_channels() - 1))
                        .iter()
                        .map(|sample| sample * scale)
                        .collect();
                    let mut fft = FFTConvolver::default();
                    fft.init(1024, &scaled).unwrap();
                    Convolver::Fft(Box::new(fft))
                })
                .collect();
            node.registration.post_message(ConvolverInfosMessage {
                convolvers: Some(convolvers),
                impulse_length: impulse.length(),
                impulse_number_of_channels: impulse.number_of_channels(),
            });
        }
        node
    }

    fn render_node(
        input: Vec<Vec<f32>>,
        impulse: Vec<Vec<f32>>,
        normalize: bool,
        force_fft: bool,
    ) -> AudioBuffer {
        let mut context = OfflineAudioContext::new(2, 4096, 48000.);
        let conv = node_with_backend(
            &context,
            AudioBuffer::from(impulse, 48000.),
            normalize,
            force_fft,
        );
        let mut src = context.create_buffer_source();
        src.set_buffer(AudioBuffer::from(input, 48000.));
        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();
        context.start_rendering_sync()
    }

    #[test]
    fn test_sparse_node_routing_normalization_and_mixed_backends() {
        for inputs in [1, 2] {
            for ir_channels in [2, 4] {
                for normalize in [false, true] {
                    for mixed in [false, true] {
                        let input: Vec<Vec<f32>> = (0..inputs)
                            .map(|channel| {
                                (0..1536)
                                    .map(|frame| ((frame * 17 + channel * 31) as f32 * 0.13).sin())
                                    .collect()
                            })
                            .collect();
                        let impulse: Vec<Vec<f32>> = (0..ir_channels)
                            .map(|channel| {
                                let mut ir = vec![0.; 1027];
                                ir[3 + channel] = 0.625;
                                ir[128 + channel] = -0.25;
                                ir[1026 - channel] = 0.125;
                                if mixed && channel % 2 == 1 {
                                    for tap in 0..9 {
                                        ir[256 + tap * 7] = 0.03125;
                                    }
                                }
                                assert_eq!(
                                    matches!(Convolver::new(1024, &ir), Convolver::Sparse(_)),
                                    !mixed || channel % 2 == 0
                                );
                                ir
                            })
                            .collect();
                        let sparse = render_node(input.clone(), impulse.clone(), normalize, false);
                        let fft = render_node(input.clone(), impulse.clone(), normalize, true);
                        let scale = if normalize {
                            normalize_buffer(&AudioBuffer::from(impulse.clone(), 48000.))
                        } else {
                            1.
                        };
                        for output_channel in 0..2 {
                            let actual = sparse.get_channel_data(output_channel);
                            let reference = fft.get_channel_data(output_channel);
                            let peak = actual
                                .iter()
                                .fold(0_f32, |peak, sample| peak.max(sample.abs()));
                            for frame in 0..actual.len() {
                                let routes = if ir_channels == 4 {
                                    vec![output_channel, output_channel + 2]
                                } else {
                                    vec![output_channel]
                                };
                                let expected: f32 = routes
                                    .iter()
                                    .map(|&ir_channel| {
                                        let source = if inputs == 1 {
                                            0
                                        } else if ir_channels == 4 {
                                            ir_channel / 2
                                        } else {
                                            ir_channel
                                        };
                                        impulse[ir_channel]
                                            .iter()
                                            .enumerate()
                                            .filter(|(tap, gain)| {
                                                *tap <= frame
                                                    && frame - *tap < input[source].len()
                                                    && **gain != 0.
                                            })
                                            .map(|(tap, gain)| {
                                                input[source][frame - tap] * (gain * scale)
                                            })
                                            .sum::<f32>()
                                    })
                                    .sum();
                                assert!(
                                    (actual[frame] - expected).abs() <= peak * 1e-6,
                                    "routing {inputs}->{ir_channels}, frame {frame}"
                                );
                                assert!(
                                    (actual[frame] - reference[frame]).abs() <= peak * 1e-6,
                                    "FFT alignment at {frame}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_sparse_node_set_buffer_cuts_old_tail() {
        for force_fft in [false, true] {
            let mut context = OfflineAudioContext::new(1, 1024, 48000.);
            let mut old_ir = vec![0.; 512];
            old_ir[0] = 1.;
            old_ir[511] = 0.5;
            let mut conv = node_with_backend(
                &context,
                AudioBuffer::from(vec![old_ir], 48000.),
                false,
                force_fft,
            );
            let mut input = vec![0.; 385];
            input[0] = 1.;
            input[384] = 0.75;
            let mut src = context.create_buffer_source();
            src.set_buffer(AudioBuffer::from(vec![input], 48000.));
            src.connect(&conv);
            conv.connect(&context.destination());
            src.start();
            context.suspend_sync(256. / 48000., move |_| {
                conv.set_buffer(AudioBuffer::from(vec![vec![0., 0.25, 0.]], 48000.));
            });
            let result = context.start_rendering_sync();
            for (frame, &sample) in result.get_channel_data(0).iter().enumerate() {
                let expected = match frame {
                    0 => 1.,
                    385 => 0.1875,
                    _ => 0.,
                };
                assert!((sample - expected).abs() < 1e-6, "stale tail at {frame}");
            }
        }
    }

    #[test]
    fn test_sparse_node_tail_ends_at_last_frame() {
        for length in [127, 128, 129, 1027] {
            let mut ir = vec![0.; length];
            ir[0] = 0.625;
            ir[length - 1] = 0.125;
            let result = render_node(vec![vec![1.]], vec![ir.clone(), ir], false, false);
            for channel in 0..2 {
                let output = result.get_channel_data(channel);
                assert_eq!(output[0], 0.625);
                assert_eq!(output[length - 1], 0.125);
                assert!(output[1..length - 1].iter().all(|sample| *sample == 0.));
                assert!(output[length..].iter().all(|sample| *sample == 0.));
            }
        }
    }

    #[test]
    #[should_panic]
    fn test_buffer_sample_rate_matches() {
        let context = OfflineAudioContext::new(1, 128, 44100.);

        let ir = vec![1.];
        let ir = AudioBuffer::from(vec![ir; 1], 48000.); // sample_rate differs
        let options = ConvolverOptions {
            buffer: Some(ir),
            ..ConvolverOptions::default()
        };

        let _ = ConvolverNode::new(&context, options);
    }

    #[test]
    #[should_panic]
    fn test_buffer_must_have_1_2_4_channels() {
        let context = OfflineAudioContext::new(1, 128, 48000.);

        let ir = vec![1.];
        let ir = AudioBuffer::from(vec![ir; 3], 48000.); // three channels
        let options = ConvolverOptions {
            buffer: Some(ir),
            ..ConvolverOptions::default()
        };

        let _ = ConvolverNode::new(&context, options);
    }

    #[test]
    fn test_constructor_options_buffer() {
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(1, 10, sample_rate);

        let ir = vec![1.];
        let calibration = 0.00125;
        let channel_data = vec![0., 1., 0., -1., 0.];
        let expected = [0., calibration, 0., -calibration, 0., 0., 0., 0., 0., 0.];

        // identity ir
        let ir = AudioBuffer::from(vec![ir; 1], sample_rate);
        let options = ConvolverOptions {
            buffer: Some(ir),
            ..ConvolverOptions::default()
        };
        let conv = ConvolverNode::new(&context, options);
        conv.connect(&context.destination());

        let buffer = AudioBuffer::from(vec![channel_data; 1], sample_rate);
        let mut src = context.create_buffer_source();
        src.connect(&conv);
        src.set_buffer(buffer);
        src.start();

        let output = context.start_rendering_sync();

        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    fn test_convolve(signal: &[f32], impulse_resp: Option<Vec<f32>>, length: usize) -> AudioBuffer {
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(1, length, sample_rate);

        let input = AudioBuffer::from(vec![signal.to_vec()], sample_rate);
        let mut src = AudioBufferSourceNode::new(&context, AudioBufferSourceOptions::default());
        src.set_buffer(input);
        src.start();

        let mut conv = ConvolverNode::new(&context, ConvolverOptions::default());
        if let Some(ir) = impulse_resp {
            conv.set_buffer(AudioBuffer::from(vec![ir.to_vec()], sample_rate));
        }

        src.connect(&conv);
        conv.connect(&context.destination());

        context.start_rendering_sync()
    }

    #[test]
    fn test_passthrough() {
        let output = test_convolve(&[0., 1., 0., -1., 0.], None, 10);
        let expected = [0., 1., 0., -1., 0., 0., 0., 0., 0., 0.];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_empty() {
        let ir = vec![];
        let output = test_convolve(&[0., 1., 0., -1., 0.], Some(ir), 10);
        let expected = [0.; 10];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_zeroed() {
        let ir = vec![0., 0., 0., 0., 0., 0.];
        let output = test_convolve(&[0., 1., 0., -1., 0.], Some(ir), 10);
        let expected = [0.; 10];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_identity() {
        let ir = vec![1.];
        let calibration = 0.00125;
        let output = test_convolve(&[0., 1., 0., -1., 0.], Some(ir), 10);
        let expected = [0., calibration, 0., -calibration, 0., 0., 0., 0., 0., 0.];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_two_id() {
        let ir = vec![1., 1.];
        let calibration = 0.00125;
        let output = test_convolve(&[0., 1., 0., -1., 0.], Some(ir), 10);
        let expected = [
            0.,
            calibration,
            calibration,
            -calibration,
            -calibration,
            0.,
            0.,
            0.,
            0.,
            0.,
        ];
        assert_float_eq!(output.get_channel_data(0), &expected[..], abs_all <= 1E-6);
    }

    #[test]
    fn test_should_have_tail_time() {
        // impulse response of length 256
        const IR_LEN: usize = 256;
        let ir = vec![1.; IR_LEN];

        // unity input signal
        let input = &[1.];

        // render into a buffer of size 512
        let output = test_convolve(input, Some(ir), 512);

        // we expect non-zero output in the range 0 to IR_LEN
        let output = output.channel_data(0).as_slice();
        assert!(!output[..IR_LEN].iter().any(|v| *v <= 1E-6));
        assert_float_eq!(&output[IR_LEN..], &[0.; 512 - IR_LEN][..], abs_all <= 1E-6);
    }

    #[test]
    fn test_channel_config_1_chan_in_1_chan_ir() {
        let number_of_channels = 1;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1.]], sample_rate);
        let ir = AudioBuffer::from(vec![vec![0., 1.]], sample_rate);

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected = [0.; 128];
        expected[1] = 1.;

        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_1_chan_in_2_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1.]], sample_rate);
        let ir = AudioBuffer::from(vec![vec![0., 1., 0.], vec![0., 0., 1.]], sample_rate);

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[2] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_2_chan_in_1_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1., 0.], vec![0., 1.]], sample_rate);
        let ir = AudioBuffer::from(vec![vec![0., 1.]], sample_rate);

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[2] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_2_chan_in_2_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1., 0.], vec![0., 1.]], sample_rate);
        let ir = AudioBuffer::from(vec![vec![0., 1., 0.], vec![0., 0., 1.]], sample_rate);

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[3] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_2_chan_in_4_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1., 0.], vec![0., 1.]], sample_rate);
        let ir = AudioBuffer::from(
            vec![
                vec![0., 1., 0., 0., 0.], // in 0 -> out 0
                vec![0., 0., 1., 0., 0.], // in 0 -> out 1
                vec![0., 0., 0., 1., 0.], // in 1 -> out 0
                vec![0., 0., 0., 0., 1.], // in 1 -> out 1
            ],
            sample_rate,
        );

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;
        expected_left[4] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[2] = 1.;
        expected_right[5] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }

    #[test]
    fn test_channel_config_1_chan_in_4_chan_ir() {
        let number_of_channels = 2;
        let length = 128;
        let sample_rate = 44100.;
        let mut context = OfflineAudioContext::new(number_of_channels, length, sample_rate);

        let input = AudioBuffer::from(vec![vec![1., 0.]], sample_rate);
        let ir = AudioBuffer::from(
            vec![
                vec![0., 1., 0., 0., 0.], // in 0 -> out 0
                vec![0., 0., 1., 0., 0.], // in 0 -> out 1
                vec![0., 0., 0., 1., 0.], // in 0 -> out 0
                vec![0., 0., 0., 0., 1.], // in 0 -> out 1
            ],
            sample_rate,
        );

        let mut src = AudioBufferSourceNode::new(
            &context,
            AudioBufferSourceOptions {
                buffer: Some(input),
                ..AudioBufferSourceOptions::default()
            },
        );

        let conv = ConvolverNode::new(
            &context,
            ConvolverOptions {
                buffer: Some(ir),
                disable_normalization: true,
                ..ConvolverOptions::default()
            },
        );

        src.connect(&conv);
        conv.connect(&context.destination());
        src.start();

        let result = context.start_rendering_sync();

        let mut expected_left = [0.; 128];
        expected_left[1] = 1.;
        expected_left[3] = 1.;

        let mut expected_right = [0.; 128];
        expected_right[2] = 1.;
        expected_right[4] = 1.;

        assert_eq!(result.number_of_channels(), 2);
        assert_float_eq!(
            result.get_channel_data(0)[..],
            expected_left[..],
            abs_all <= 1e-7
        );
        assert_float_eq!(
            result.get_channel_data(1)[..],
            expected_right[..],
            abs_all <= 1e-7
        );
    }
}
