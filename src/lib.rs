//! kokoro-tiny: A minimal, embeddable TTS engine using the Kokoro model
//!
//! This crate provides a simple API for text-to-speech synthesis using the
//! Kokoro 82M parameter model. Perfect for embedding in other applications!
//!
//! # Example
//! ```no_run
//! use kokoro_tiny::TtsEngine;
//!
//! #[tokio::main]
//! async fn main() {
//!     let mut tts = TtsEngine::new("af_heart").await.unwrap();
//!     let sections = vec![
//!         ("Chapter 1".to_string(), "Hello world!".to_string()),
//!     ];
//!     let (_aac, _sample_sizes, _chapters) = tts.synthesize(&sections);
//! }
//! ```

use std::collections::VecDeque;
use std::fs;
#[cfg(feature = "mp3")]
use std::fs::File;
#[cfg(feature = "mp3")]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use std::time::Duration;

use audio_sample::ConvertTo;
use candle_core::{DType, Device};
use kokoro_tts::model::{Kokoro, VoicePack};
use kokoro_tts::phonemizer::{Phonemizer as CandlePhonemizer, TwoTierPhonemizer};

#[cfg(feature = "playback")]
use rodio::{Decoder, OutputStream, Sink};
#[cfg(feature = "playback")]
use std::io::Cursor;

// Constants
const SAMPLE_RATE: u32 = 24000;
//const DEFAULT_VOICE: &str = "af_sky";
const DEFAULT_SPEED: f32 = 1.0;

// Get cache directory for shared model storage (Hue's suggestion!)
fn get_cache_dir() -> PathBuf {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".cache").join("kokoros").join("candle")
}

pub struct Phonemizer {
    inner: TwoTierPhonemizer,
}

impl Phonemizer {
    pub fn new() -> Phonemizer {
        Phonemizer {
            inner: TwoTierPhonemizer,
        }
    }

    fn split_for_model(phonemes: &str, max_phonemes: usize) -> Vec<String> {
        let mut remaining: Vec<char> = phonemes.chars().collect();
        let mut chunks = Vec::new();

        while remaining.len() > max_phonemes {
            let split_at = remaining[..max_phonemes]
                .iter()
                .rposition(|c| matches!(c, ';' | ':' | ',' | '.' | '!' | '?' | '—' | '-' | ' '))
                .map(|idx| idx + 1)
                .filter(|idx| *idx >= max_phonemes / 2)
                .unwrap_or(max_phonemes);
            let chunk: String = remaining.drain(..split_at).collect();
            let chunk = chunk.trim().to_string();
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
        }

        let tail: String = remaining.into_iter().collect();
        let tail = tail.trim().to_string();
        if !tail.is_empty() {
            chunks.push(tail);
        }
        chunks
    }

    pub fn graphemes_to_phonemes<'a>(
        &self,
        text: &'a str,
        _use_espeak: bool,
    ) -> Option<(&'a str, VecDeque<String>)> {
        let sentence_phonemes = match self.inner.phonemize_chunks(text) {
            Ok(chunks) => chunks,
            Err(error) => {
                eprintln!("Warning: Unable to phonemize text: {error}");
                return None;
            }
        };

        let chunks: VecDeque<String> = sentence_phonemes
            .into_iter()
            .flat_map(|phonemes| Self::split_for_model(&phonemes, 510))
            .collect();
        if chunks.is_empty() {
            println!("Warning: Empty Text: {text}");
            return None;
        }
        Some((text, chunks))
    }
}

struct SessionHandler {
    model: Kokoro,
    voice: VoicePack,
}

impl SessionHandler {
    fn load(model_dir: &Path, voice_path: &Path, device: Device) -> Result<Self, String> {
        let model = Kokoro::load(model_dir, &device)
            .map_err(|e| format!("Failed to load Candle Kokoro model: {e}"))?;
        let voice = VoicePack::load(voice_path, &device)
            .map_err(|e| format!("Failed to load Candle Kokoro voice: {e}"))?;
        Ok(Self { model, voice })
    }

    pub fn inference(&mut self, phonemes: &str, speed: f32) -> Result<Vec<f32>, String> {
        let phoneme_count = self.model.phonemes_to_ids(phonemes).len();
        let style = self
            .voice
            .style(phoneme_count)
            .map_err(|e| format!("Failed to select voice style: {e}"))?;
        self.model
            .forward(phonemes, &style, f64::from(speed))
            .and_then(|audio| audio.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>())
            .map_err(|e| format!("Candle Kokoro inference failed: {e}"))
    }
}

struct Chunk {
    data: Vec<f32>,
}

impl Chunk {
    fn new() -> Chunk {
        Chunk { data: Vec::new() }
    }
}

pub struct Paragraph {
    finished_chunks: usize,
    // These need to be joined with internal trimming, so don't
    // trim the start on the first chunk, or the end on the last chunk.
    chunks: Vec<Chunk>,
}

impl Paragraph {
    fn new() -> Paragraph {
        Paragraph {
            finished_chunks: 0,
            chunks: Vec::new(),
        }
    }

    fn combine_chunks(&self) -> Vec<f32> {
        let mut data: Vec<f32> = Vec::new();

        match self.chunks.len() {
            0 => return Vec::new(),
            1 => return self.chunks[0].data.clone(),
            _ => {
                for (i, chunk) in self.chunks.iter().enumerate() {
                    let trim_type = if i == 0 {
                        TrimSection::End
                    } else if i + 1 == self.chunks.len() {
                        TrimSection::Begin
                    } else {
                        TrimSection::BeginAndEnd
                    };

                    data.extend(
                        trim_with_auto_ref(&chunk.data, trim_type, 60.0, 2048, 512).unwrap(),
                    );
                }
            }
        }

        data
    }
}

struct Section {
    finished_paragraphs: usize,
    paragraphs: Vec<Paragraph>, // These can be cleanly joined.
}

impl Section {
    fn new() -> Section {
        Section {
            finished_paragraphs: 0,
            paragraphs: Vec::new(),
        }
    }
}

pub trait Encoder {
    fn feed_audio(&mut self, paragraph: &mut Paragraph, finished: bool) -> usize;
}

pub struct AACEncoder {
    remaining_samples: Vec<i16>,
    encoder: fdk_aac::enc::Encoder,
    encoder_info: fdk_aac::enc::InfoStruct,
    temp_output: Vec<u8>,
    final_output: Vec<u8>,
    sample_sizes: Vec<u16>,
}

pub type AACEncoderBitrate = fdk_aac::enc::BitRate;

impl AACEncoder {
    pub fn new(bit_rate: AACEncoderBitrate) -> AACEncoder {
        let params = fdk_aac::enc::EncoderParams {
            bit_rate: bit_rate,
            //bit_rate: fdk_aac::enc::BitRate::Cbr(24000),
            sample_rate: SAMPLE_RATE,
            transport: fdk_aac::enc::Transport::Raw,
            //transport: fdk_aac::enc::Transport::Adts,
            channels: fdk_aac::enc::ChannelMode::Mono,
            audio_object_type: fdk_aac::enc::AudioObjectType::Mpeg4LowComplexity,
        };

        let encoder = fdk_aac::enc::Encoder::new(params).unwrap();
        let encoder_info: fdk_aac::enc::InfoStruct = encoder.info().unwrap();

        AACEncoder {
            remaining_samples: Vec::new(),
            encoder,
            encoder_info,
            temp_output: vec![0; (6144 / 8) * 1 /* channels */],
            final_output: Vec::new(),
            sample_sizes: Vec::new(),
        }
    }
}

impl Encoder for AACEncoder {
    fn feed_audio(&mut self, paragraph: &mut Paragraph, finished: bool) -> usize {
        let converted_audio: Vec<i16> = paragraph
            .combine_chunks()
            .into_iter()
            .map(|s| s.convert_to())
            .collect();
        paragraph.chunks.clear();

        let samples_grabbed = converted_audio.len();
        self.remaining_samples.extend(converted_audio);

        loop {
            if !finished && self.remaining_samples.len() < (self.encoder_info.frameLength as usize)
            {
                return samples_grabbed;
            }

            let frames_to_send =
                (self.encoder_info.frameLength as usize).min(self.remaining_samples.len());

            let encoding_info = self
                .encoder
                .encode(
                    &self.remaining_samples[0..frames_to_send],
                    &mut self.temp_output,
                )
                .unwrap();

            self.final_output
                .extend(&self.temp_output[0..encoding_info.output_size]);
            self.sample_sizes.push(encoding_info.output_size as u16);
            self.remaining_samples
                .drain(0..encoding_info.input_consumed);
        }
    }
}

/// Main TTS engine struct
pub struct TtsEngine {
    sessions: Vec<SessionHandler>,
    phonemizer: Phonemizer,
}

impl TtsEngine {
    /// Create a new TTS engine, downloading model files if necessary
    /// Uses KOKORO_MODEL_DIR or ~/.cache/kokoros/candle for model storage.
    pub async fn new(voice: &str) -> Result<Self, String> {
        let model_dir = std::env::var_os("KOKORO_MODEL_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(get_cache_dir);
        let voice_path = model_dir
            .join("voices")
            .join(format!("{voice}.safetensors"));

        Self::with_paths(
            model_dir
                .to_str()
                .ok_or_else(|| "Model path is not valid UTF-8".to_string())?,
            voice_path
                .to_str()
                .ok_or_else(|| "Voice path is not valid UTF-8".to_string())?,
        )
        .await
    }

    fn worker_count(name: &str, default: usize) -> Result<usize, String> {
        match std::env::var(name) {
            Ok(value) => value
                .parse()
                .map_err(|e| format!("Invalid {name} value {value:?}: {e}")),
            Err(std::env::VarError::NotPresent) => Ok(default),
            Err(error) => Err(format!("Unable to read {name}: {error}")),
        }
    }

    fn create_sessions(model_dir: &Path, voice_path: &Path) -> Result<Vec<SessionHandler>, String> {
        let mut sessions = Vec::new();

        #[cfg(feature = "cuda")]
        {
            let cuda_workers = Self::worker_count("KOKORO_CUDA_WORKERS", 1)?;
            for worker in 0..cuda_workers {
                let device = Device::new_cuda(0)
                    .map_err(|e| format!("Failed to initialize CUDA worker {worker}: {e}"))?;
                sessions.push(SessionHandler::load(model_dir, voice_path, device)?);
            }
        }

        let cpu_workers = Self::worker_count("KOKORO_CPU_WORKERS", 1)?;
        for _ in 0..cpu_workers {
            sessions.push(SessionHandler::load(model_dir, voice_path, Device::Cpu)?);
        }

        if sessions.is_empty() {
            return Err("At least one Kokoro worker must be configured".to_string());
        }
        Ok(sessions)
    }

    /// Create a new TTS engine with custom model paths
    pub async fn with_paths(model_dir: &str, voice_path: &str) -> Result<Self, String> {
        let model_dir = Path::new(model_dir);
        let voice_path = Path::new(voice_path);
        if !model_dir.join("model.safetensors").is_file() {
            return Err(format!(
                "Missing converted Candle weights: {}",
                model_dir.join("model.safetensors").display()
            ));
        }
        if !model_dir.join("config.json").is_file() {
            return Err(format!(
                "Missing Kokoro config: {}",
                model_dir.join("config.json").display()
            ));
        }
        if !voice_path.is_file() {
            return Err(format!("Missing converted voice: {}", voice_path.display()));
        }

        let sessions = Self::create_sessions(model_dir, voice_path)?;

        Ok(Self {
            sessions,
            phonemizer: Phonemizer::new(),
        })
    }

    fn infer_thread(
        inference_queue: Arc<lockfree::queue::Queue<(usize, usize, usize, String)>>,
        audios_destination: Arc<Mutex<Vec<Section>>>,
        session: SessionHandler,
        finished_queueing: Arc<AtomicBool>,
    ) -> SessionHandler {
        let audios_destination = audios_destination;
        let mut session = session;

        let mut item: Option<(usize, usize, usize, String)> = inference_queue.pop();

        while !finished_queueing.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some((i, j, k, to_infer)) = item {
                let audio = session.inference(&to_infer, DEFAULT_SPEED).unwrap();

                println!("({i}, {j}, {k})");
                let mut destination = audios_destination.lock().unwrap();
                let section = &mut destination[i];
                let paragraph = &mut section.paragraphs[j];
                paragraph.chunks[k].data = audio;
                paragraph.finished_chunks += 1;

                if paragraph.finished_chunks == paragraph.chunks.len() {
                    section.finished_paragraphs += 1;
                    println!("Finished paragraph ({i}, {j})");
                }
            } else {
                std::thread::sleep(Duration::from_millis(1000));
            }

            item = inference_queue.pop();
        }

        println!("Finished Thread");

        return session;
    }

    /// Synthesize speech from text
    pub fn synthesize(
        &mut self,
        sections: &Vec<(String, String)>,
    ) -> (Vec<u8>, Vec<u16>, Vec<(usize, String)>) {
        let inference_queue: Arc<lockfree::queue::Queue<(usize, usize, usize, String)>> =
            Arc::new(lockfree::queue::Queue::new());

        let audios_destination: Arc<Mutex<Vec<Section>>> = Arc::new(Mutex::new(Vec::new()));

        let finished_queueing: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();

        for _ in 0..self.sessions.len() {
            let audios_destination = audios_destination.clone();
            let queue = inference_queue.clone();
            let session = self.sessions.pop().unwrap();
            let finished_queueing = finished_queueing.clone();
            handles.push(std::thread::spawn(move || {
                return Self::infer_thread(queue, audios_destination, session, finished_queueing);
            }));
        }

        {
            let mut i = 0;
            let mut j = 0;

            for section in sections {
                {
                    let mut destination = audios_destination.lock().unwrap();
                    destination.push(Section::new());
                }

                let mut paragraphs: VecDeque<(&str, VecDeque<String>)> = section
                    .1
                    .lines()
                    .filter_map(|t| self.phonemizer.graphemes_to_phonemes(t, true))
                    .collect();

                for paragraph in &mut paragraphs {
                    {
                        let mut destination = audios_destination.lock().unwrap();
                        destination[i].paragraphs.push(Paragraph::new());
                    }

                    for k in 0..paragraph.1.len() {
                        {
                            let mut destination = audios_destination.lock().unwrap();
                            destination[i].paragraphs[j].chunks.push(Chunk::new());
                        }

                        let chunk = paragraph.1.pop_front().unwrap();
                        if chunk.chars().count() > 510 {
                            println!("Oh no: \n{}", paragraph.0);
                        }

                        inference_queue.push((i, j, k, chunk));
                    }

                    j += 1;
                }

                j = 0;
                i += 1;
            }
        }

        std::thread::sleep(Duration::from_millis(200));

        let mut duration_so_far: usize = 0;
        let mut chapter_markers: Vec<(usize, String)> = Vec::new();
        let mut encoder = AACEncoder::new(fdk_aac::enc::BitRate::VbrMedium);
        for i in 0..sections.len() {
            std::thread::sleep(Duration::from_millis(1000));
            let mut copied_audio = false;
            let mut paragraph_index = 0;

            chapter_markers.push((duration_so_far, sections[i].0.clone()));

            while !copied_audio {
                std::thread::sleep(Duration::from_millis(1000));
                let generated_audios = audios_destination.clone();

                {
                    let mut generated_audios = generated_audios.lock().unwrap();

                    let length = generated_audios[i].paragraphs.len();
                    for generated_paragraph in
                        &mut generated_audios[i].paragraphs[paragraph_index..length]
                    {
                        if generated_paragraph.finished_chunks != generated_paragraph.chunks.len() {
                            break;
                        }

                        paragraph_index += 1;
                        copied_audio = paragraph_index == length;

                        duration_so_far += encoder
                            .feed_audio(generated_paragraph, copied_audio && i == sections.len());
                    }
                }

                //if generated_audios[i].finished_paragraphs != generated_audios[i].paragraphs.len() {
                //    continue;
                //}
                //audios.push(generated_audios[i].paragraphs.iter().map(|f| f.combine_chunks()).flatten().collect());
                //generated_audios[i].paragraphs.clear();
                //copied_audio = true
            }

            println!("Finished section {i}");

            //self.save_wav(&format!("audio_{i}.wav"), &audios.last().unwrap()).unwrap();
            std::fs::write(&format!("audio_{i}.txt"), sections[i].1.clone()).unwrap();
        }

        finished_queueing.store(true, std::sync::atomic::Ordering::SeqCst);

        for thread in handles {
            self.sessions.push(thread.join().unwrap());
        }

        return (encoder.final_output, encoder.sample_sizes, chapter_markers);
    }

    /// Play audio directly to the default audio device with volume control
    #[cfg(feature = "playback")]
    pub fn play(&self, audio: &[f32], volume: f32) -> Result<(), String> {
        // Convert audio to WAV format in memory
        let wav_data = self.to_wav_bytes(audio)?;

        // Setup audio output
        let (_stream, stream_handle) = OutputStream::try_default()
            .map_err(|e| format!("Failed to get audio output: {}", e))?;

        let sink = Sink::try_new(&stream_handle)
            .map_err(|e| format!("Failed to create audio sink: {}", e))?;

        // Decode WAV data from memory
        let cursor = Cursor::new(wav_data);
        let source = Decoder::new(cursor).map_err(|e| format!("Failed to decode audio: {}", e))?;

        // Set volume (0.0 to 1.0)
        sink.set_volume(volume.clamp(0.0, 1.0));

        // Play the audio
        sink.append(source);
        sink.sleep_until_end();

        Ok(())
    }

    /// List available audio devices
    #[cfg(feature = "playback")]
    pub fn list_devices() -> Vec<String> {
        use cpal::traits::{DeviceTrait, HostTrait};

        if let Ok(devices) = cpal::default_host().output_devices() {
            devices.filter_map(|device| device.name().ok()).collect()
        } else {
            vec!["default".to_string()]
        }
    }

    /// Convert audio to WAV bytes (for playback)
    #[cfg(feature = "playback")]
    fn to_wav_bytes(&self, audio: &[f32]) -> Result<Vec<u8>, String> {
        let mut buffer = Vec::new();
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        {
            let mut writer = hound::WavWriter::new(Cursor::new(&mut buffer), spec)
                .map_err(|e| format!("Failed to create WAV writer: {}", e))?;

            for &sample in audio {
                //let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
                writer
                    .write_sample(sample)
                    .map_err(|e| format!("Failed to write sample: {}", e))?;
            }

            writer
                .finalize()
                .map_err(|e| format!("Failed to finalize WAV: {}", e))?;
        }

        Ok(buffer)
    }

    /// Save audio to WAV file
    pub fn save_wav(&self, path: &str, audio: &[f32]) -> Result<(), String> {
        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: SAMPLE_RATE,
            bits_per_sample: 32,
            sample_format: hound::SampleFormat::Float,
        };

        let mut writer = hound::WavWriter::create(path, spec)
            .map_err(|e| format!("Failed to create WAV file: {}", e))?;

        // Convert float samples to i16
        for &sample in audio {
            //let sample_i16 = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(sample)
                .map_err(|e| format!("Failed to write sample: {}", e))?;
        }

        writer
            .finalize()
            .map_err(|e| format!("Failed to finalize WAV: {}", e))?;
        Ok(())
    }

    #[cfg(feature = "mp3")]
    /// Save audio to MP3 file (requires mp3 feature)
    pub fn save_mp3(&self, path: &str, audio: &[f32]) -> Result<(), String> {
        use mp3lame_encoder::{Builder, Encoder, FlushNoGap};

        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Convert to i16 samples
        let samples: Vec<i16> = audio
            .iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        // Setup MP3 encoder
        let mut encoder = Builder::new()
            .map_err(|e| format!("Failed to create MP3 encoder: {:?}", e))?
            .set_num_channels(1)
            .map_err(|e| format!("Failed to set channels: {:?}", e))?
            .set_sample_rate(SAMPLE_RATE)
            .map_err(|e| format!("Failed to set sample rate: {:?}", e))?
            .set_brate(mp3lame_encoder::Bitrate::Kbps128)
            .map_err(|e| format!("Failed to set bitrate: {:?}", e))?
            .set_quality(mp3lame_encoder::Quality::Best)
            .map_err(|e| format!("Failed to set quality: {:?}", e))?
            .build()
            .map_err(|e| format!("Failed to build encoder: {:?}", e))?;

        let mut mp3_data = Vec::new();
        let encoded = encoder
            .encode(&samples)
            .map_err(|e| format!("Failed to encode: {:?}", e))?;
        mp3_data.extend_from_slice(&encoded);

        let encoded = encoder
            .flush::<FlushNoGap>()
            .map_err(|e| format!("Failed to flush: {:?}", e))?;
        mp3_data.extend_from_slice(&encoded);

        // Write to file
        let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        file.write_all(&mp3_data)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    #[cfg(feature = "opus-format")]
    /// Save audio to OPUS file - great for streaming and low bandwidth!
    pub fn save_opus(&self, path: &str, audio: &[f32], bitrate: i32) -> Result<(), String> {
        use audiopus::{coder::Encoder, Application, Channels, SampleRate};

        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Convert to i16 samples
        let samples: Vec<i16> = audio
            .iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        // Setup OPUS encoder (24kHz mono)
        let mut encoder = Encoder::new(SampleRate::Hz24000, Channels::Mono, Application::Audio)
            .map_err(|e| format!("Failed to create OPUS encoder: {:?}", e))?;

        // Set bitrate (typical: 24000 for speech)
        encoder
            .set_bitrate(bitrate)
            .map_err(|e| format!("Failed to set bitrate: {:?}", e))?;

        // Encode in chunks (OPUS needs specific frame sizes)
        let frame_size = 480; // 20ms at 24kHz
        let mut opus_data = Vec::new();

        for chunk in samples.chunks(frame_size) {
            let mut encoded = vec![0u8; 4000];
            let len = encoder
                .encode(chunk, &mut encoded)
                .map_err(|e| format!("Failed to encode OPUS: {:?}", e))?;
            opus_data.extend_from_slice(&encoded[..len]);
        }

        // Write to file
        let mut file = File::create(path).map_err(|e| format!("Failed to create file: {}", e))?;
        file.write_all(&opus_data)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    /// Save audio in any supported format based on file extension
    pub fn save_audio(&self, path: &str, audio: &[f32]) -> Result<(), String> {
        let extension = Path::new(path)
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("")
            .to_lowercase();

        match extension.as_str() {
            "wav" => self.save_wav(path, audio),
            #[cfg(feature = "mp3")]
            "mp3" => self.save_mp3(path, audio),
            #[cfg(feature = "opus-format")]
            "opus" | "ogg" => self.save_opus(path, audio, 24000),
            _ => Err(format!("Unsupported audio format: {}", extension)),
        }
    }
}

// Simple builder pattern for customization
pub struct TtsBuilder {
    model_dir: String,
    voice_path: Option<String>,
    voice: String,
}

impl Default for TtsBuilder {
    fn default() -> Self {
        let cache_dir = get_cache_dir();
        let voice = "af_heart".to_string();
        Self {
            model_dir: cache_dir.to_string_lossy().into_owned(),
            voice_path: None,
            voice,
        }
    }
}

impl TtsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_path(mut self, path: &str) -> Self {
        self.model_dir = path.to_string();
        self
    }

    pub fn voices_path(mut self, path: &str) -> Self {
        self.voice_path = Some(path.to_string());
        self
    }

    pub fn voice(mut self, voice: &str) -> Self {
        self.voice = voice.to_string();
        self
    }

    fn resolved_voice_path(&self) -> PathBuf {
        self.voice_path
            .as_ref()
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                Path::new(&self.model_dir)
                    .join("voices")
                    .join(format!("{}.safetensors", self.voice))
            })
    }

    pub async fn build(self) -> Result<TtsEngine, String> {
        let voice_path = self.resolved_voice_path();
        TtsEngine::with_paths(&self.model_dir, &voice_path.to_string_lossy()).await
    }
}

/*


//def amplitude_to_db(
//    S,
//    *,
//    ref: float | Callable = 1.0,
//    amin: float = 1e-5,
//    top_db: float | None = 80.0,
//) -> np.floating[Any] | np.ndarray:
fn amplitude_to_db(
    audio: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
    a_min: Option<f32>,
    top_db: Option<f32>
) -> Vec<f32> {
    let top_db = top_db.unwrap_or(80.0);
    let a_min = a_min.unwrap_or(1e-5);
//    S = np.asarray(S)

//    if np.issubdtype(S.dtype, np.complexfloating):
//        warnings.warn(
//            "amplitude_to_db was called on complex input so phase "
//            "information will be discarded. To suppress this warning, "
//            "call amplitude_to_db(np.abs(S)) instead.",
//            stacklevel=2,
//        )


//    magnitude = np.abs(S)
//
//    if callable(ref):
//        # User supplied a function to calculate reference power
//        ref_value = ref(magnitude)
//    else:
//        ref_value = np.abs(ref)
//
//    out_array = magnitude if isinstance(magnitude, np.ndarray) else None
//    power = np.square(magnitude, out=out_array)
//
//    db: np.ndarray = power_to_db(power, ref=ref_value**2, amin=amin**2, top_db=top_db)
//    return db

    let magnitude = audio.abs();
    let S = magnitude.iter().max();

    Vec::new()
}


//let test = ArrayView::from(&audio);
//let abs = test.abs();

fn _signal_to_frame_nonsilent(
    audio: ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>,
    frame_length: Option<i32>,
    hop_length: Option<i32>,
    top_db: Option<f32>
) -> Vec<f32> {
    let _ = audio;
    let frame_length = frame_length.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(512);
    let top_db = top_db.unwrap_or(60.0);
    //let ref_fn = ref_fn.unwrap_or(60.0);
    //let aggregate = aggregate.unwrap_or(60.0);


    Vec::new()
}





fn trim_audio(
    audio: &Vec<f32>,
    top_db: Option<f32>,
    //ref: float | Callable = np.max,
    frame_length: Option<i32>,
    hop_length: Option<i32>
    //aggregate: Callable = np.max,
) -> (Vec<f32>, Vec<f32>) {
    let top_db = top_db.unwrap_or(60.0);
    let frame_length = frame_length.unwrap_or(2048);
    let hop_length = hop_length.unwrap_or(512);

    let audio_view = ArrayView::from(&audio);

    let non_silent = _signal_to_frame_nonsilent(audio_view, Some(frame_length), Some(hop_length), Some(top_db));

    (Vec::new(), Vec::new())
}

 */

// Copyright (c) 2013--2023, librosa development team (Python original)
// Rust port by Copilot, 2024
//
// ***This file extracted and adapted from librosa (Python) for use as a standalone Rust module.***
//
// Reference (Python):
//     - https://gist.github.com/evq/82e95a363eeeb75d15dd62abc1eb1bde
//     - https://github.com/librosa/librosa/blob/894942673d55aa2206df1296b6c4c50827c7f1d6/librosa/effects.py#L612

use std::f32;
use std::fmt;

#[derive(Debug)]
pub struct LibrosaError(pub String);

impl fmt::Display for LibrosaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LibrosaError: {}", self.0)
    }
}

impl std::error::Error for LibrosaError {}

pub fn abs2(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v * v).collect()
}

pub fn amplitude_to_db(s: &[f32], ref_val: f32, amin: f32, top_db: Option<f32>) -> Vec<f32> {
    let magnitude: Vec<f32> = s.iter().map(|&x| x.abs()).collect();
    let power: Vec<f32> = magnitude.iter().map(|&x| x * x).collect();
    power_to_db(&power, ref_val * ref_val, amin * amin, top_db)
}

pub fn power_to_db(s: &[f32], ref_val: f32, amin: f32, top_db: Option<f32>) -> Vec<f32> {
    let log_spec: Vec<f32> = s
        .iter()
        .map(|&x| 10.0 * (x.max(amin)).log10() - 10.0 * ref_val.max(amin).log10())
        .collect();

    if let Some(top_db) = top_db {
        let max_log = log_spec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        log_spec.iter().map(|&x| x.max(max_log - top_db)).collect()
    } else {
        log_spec
    }
}

// Framing: slice a 1D array into overlapping frames
pub fn frame(
    x: &[f32],
    frame_length: usize,
    hop_length: usize,
) -> Result<Vec<Vec<f32>>, LibrosaError> {
    if x.len() < frame_length {
        return Err(LibrosaError(format!(
            "Input is too short (n={}) for frame_length={}",
            x.len(),
            frame_length
        )));
    }
    if hop_length < 1 {
        return Err(LibrosaError(format!("Invalid hop_length: {}", hop_length)));
    }
    let n_frames = 1 + (x.len() - frame_length) / hop_length;
    let mut frames = Vec::with_capacity(n_frames);
    for i in 0..n_frames {
        let start = i * hop_length;
        let end = start + frame_length;
        frames.push(x[start..end].to_vec());
    }
    Ok(frames)
}

// RMS calculation for an audio signal (1D)
pub fn rms(
    y: &[f32],
    frame_length: usize,
    hop_length: usize,
    center: bool,
) -> Result<Vec<f32>, LibrosaError> {
    let mut padded = Vec::new();
    if center {
        let pad = frame_length / 2;
        padded.extend(std::iter::repeat(0.0).take(pad));
        padded.extend_from_slice(y);
        padded.extend(std::iter::repeat(0.0).take(pad));
    } else {
        padded.extend_from_slice(y);
    }
    let frames = frame(&padded, frame_length, hop_length)?;
    Ok(frames
        .iter()
        .map(|f| {
            let mean_sq = f.iter().map(|&x| x * x).sum::<f32>() / f.len() as f32;
            mean_sq.sqrt()
        })
        .collect())
}

// Convert frame indices to sample indices
pub fn frames_to_samples(frames: &[usize], hop_length: usize, n_fft: Option<usize>) -> Vec<usize> {
    let offset = n_fft.map_or(0, |n| n / 2);
    frames.iter().map(|&f| f * hop_length + offset).collect()
}

pub enum TrimSection {
    BeginAndEnd,
    Begin,
    End,
}

// Core trim function
pub fn trim(
    y: &[f32],
    trim_type: TrimSection,
    top_db: f32,
    ref_val: f32,
    frame_length: usize,
    hop_length: usize,
) -> Result<Vec<f32>, LibrosaError> {
    let rms_vals = rms(y, frame_length, hop_length, true)?;
    let db = amplitude_to_db(&rms_vals, ref_val, 1e-5, None);
    let non_silent: Vec<bool> = db.iter().map(|&d| d > -top_db).collect();
    let nonzero: Vec<usize> = non_silent
        .iter()
        .enumerate()
        .filter_map(|(i, &val)| if val { Some(i) } else { None })
        .collect();

    let (start, end) = if !nonzero.is_empty() {
        let start = frames_to_samples(&[nonzero[0]], hop_length, None)[0];
        let end = {
            let e = frames_to_samples(&[nonzero[nonzero.len() - 1] + 1], hop_length, None)[0];
            e.min(y.len())
        };
        (start, end)
    } else {
        (0, 0)
    };

    let (start, end) = match trim_type {
        TrimSection::Begin => (start, y.len()),
        TrimSection::BeginAndEnd => (start, end),
        TrimSection::End => (0, end),
    };

    Ok(y[start..end].to_vec())
}

// Optional: helper for auto ref_val as max (librosa default)
pub fn trim_with_auto_ref(
    y: &[f32],
    trim_type: TrimSection,
    top_db: f32,
    frame_length: usize,
    hop_length: usize,
) -> Result<Vec<f32>, LibrosaError> {
    let ref_val = y.iter().cloned().fold(f32::NEG_INFINITY, f32::max).abs();
    trim(y, trim_type, top_db, ref_val, frame_length, hop_length)
}

// // Tests
// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn test_trim_simple() {
//         // Silence, then tone, then silence
//         let mut y = vec![0.0; 100];
//         y.extend(vec![1.0; 500]);
//         y.extend(vec![0.0; 100]);
//         let (y_trimmed, idx) = trim_with_auto_ref(
//             &y,
//             60.0,
//             2048,
//             512
//         ).unwrap();
//         assert!(!y_trimmed.is_empty());
//         assert!(y_trimmed.iter().all(|&v| v == 1.0));
//         assert_eq!(y_trimmed.len(), idx[1] - idx[0]);
//     }
// }

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_engine_creation() {
        // This would need the model files to be present
        // let engine = TtsEngine::new().await;
        // assert!(engine.is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let builder = TtsBuilder::new().model_path("models").voice("am_michael");
        assert_eq!(
            builder.resolved_voice_path(),
            Path::new("models")
                .join("voices")
                .join("am_michael.safetensors")
        );

        let explicit_path = Path::new("custom").join("voice.safetensors");
        let builder = builder.voices_path(&explicit_path.to_string_lossy());
        assert_eq!(builder.resolved_voice_path(), explicit_path);
    }

    #[test]
    fn phoneme_chunks_respect_model_context() {
        let phonemes = format!("{} {}", "a".repeat(700), "b".repeat(400));
        let chunks = Phonemizer::split_for_model(&phonemes, 510);
        assert!(chunks.len() >= 3);
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 510));
    }

    #[test]
    fn candle_phonemizer_produces_model_chunks() {
        let phonemizer = Phonemizer::new();
        let (_, chunks) = phonemizer
            .graphemes_to_phonemes("Candle handles the grapheme to phoneme conversion.", true)
            .expect("text should phonemize");
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|chunk| chunk.chars().count() <= 510));
    }
}
