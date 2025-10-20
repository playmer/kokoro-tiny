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
//!     // Initialize with auto-download of model if needed
//!     let mut tts = TtsEngine::new().await.unwrap();
//!
//!     // Generate speech
//!     let audio = tts.synthesize("Hello world!", None).unwrap();
//!
//!     // Save to file
//!     tts.save_wav("output.wav", &audio).unwrap();
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider, ROCmExecutionProvider};
use ort::value::TensorValueType;
use unicode_segmentation::UnicodeSegmentation;
use espeak_rs::{text_to_phonemes, _text_to_phonemes};
use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionInputs, SessionInputValue},
    value::{Tensor, Value},
};

#[cfg(feature = "playback")]
use rodio::{Decoder, OutputStream, Sink, Source};
use std::io::Cursor;

// Constants
const MODEL_URL: &str = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx";
const VOICES_URL: &str = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin";
const SAMPLE_RATE: u32 = 24000;
const DEFAULT_VOICE: &str = "af_sky";
const DEFAULT_SPEED: f32 = 1.0;

// Get cache directory for shared model storage (Hue's suggestion!)
fn get_cache_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    Path::new(&home).join(".cache").join("kokoros")
}
    
//static PHONETISAURUS_MODEL: &[u8] = include_bytes!("model.fst");

pub struct Phonemizer {
    //phonetisaurus_phonemizer: phonetisaurus_g2p::PhonetisaurusModel
    vocab: HashMap<char, i64>,
}

struct OffsetAndSize {
    offset: usize,
    size: usize
}

enum WordOrNonWord {
    Word(OffsetAndSize),
    NonWord(OffsetAndSize),
}

impl Phonemizer {
    fn get_whitespace_and_words(text: &str) -> Vec<WordOrNonWord> {
        let mut tokens = Vec::new();

        let words: Vec<(usize, &str)> = text.unicode_word_indices().collect();

        for i in 0..words.len() {
            let (offset, slice) = words[i];

            tokens.push(WordOrNonWord::Word(OffsetAndSize{offset, size: slice.len()}));

            if (i + 1) < words.len() {
                let (next_word_offset, _) = words[i + 1];
                let end_of_word = offset + slice.len();
                tokens.push(WordOrNonWord::NonWord(OffsetAndSize{offset: offset + slice.len(), size: next_word_offset - end_of_word}));
            }
        }

        return tokens;
    }


    pub fn new() -> Phonemizer {
        Phonemizer {
            //phonetisaurus_phonemizer: phonetisaurus_g2p::PhonetisaurusModel::try_from(PHONETISAURUS_MODEL).unwrap()
            vocab: build_vocab()
        }
    }

    

    fn tokenize_phonemes(&self, phonemes: &str) -> Vec<i64> {
        // Use proper vocabulary-based tokenization like original Kokoros
        let tokens: Vec<i64> = phonemes
            .chars()
            .filter_map(|c| self.vocab.get(&c).copied())
            .collect();

        // Return as a batch of one sequence
        tokens
    }

    fn split_index(phonemes: &mut Vec<String>, i: usize, c: char) -> bool {
        let current_string = phonemes[i].clone();

        let split: Vec<&str> = current_string.splitn(2, c).collect();
        if split.len() == 2 {
            phonemes[i] = split[0].to_string();
            phonemes.insert(i + 1, split[1].to_string());
            return true;
        }

        return false;
    }

    
    pub fn graphemes_to_phonemes(&self, text: &str, use_espeak: bool) -> VecDeque<Vec<i64>>  {
        let mut phonemes = Vec::new();

        phonemes.push(_text_to_phonemes(text.trim(), "en-us", None, true, false).unwrap().join(""));

        let mut i = 0;

        while phonemes[i].len() > 510 {
            if Self::split_index(&mut phonemes, i, ';') && phonemes[i].len() > 510 {
                continue;
            }
            
            if phonemes[i].len() > 510 && Self::split_index(&mut phonemes, i, ',') && phonemes[i].len() > 510 {
                continue;
            }

            if phonemes[i].len() > 510 && Self::split_index(&mut phonemes, i, '-') && phonemes[i].len() > 510 {
                continue;
            }

            if phonemes[i].len() > 510 && Self::split_index(&mut phonemes, i, ' ') && phonemes[i].len() > 510 {
                continue;
            }

            i += 1;
        }

        phonemes.iter().map(|f| self.tokenize_phonemes(f)).collect()
    }

//pub fn graphemes_to_phonemes(&self, text: &str, use_espeak: bool) -> String  {
//    let text= text.trim();
//    let mut espeak_phonemes = String::new();
//    let mut phonetisaurus_phonemes = String::new();
//    
//    let mut combined_phonemes = String::new();
//    for word_or_nonword in Self::get_whitespace_and_words(text) {
//        match word_or_nonword {
//            WordOrNonWord::NonWord(value) => {
//                let nonword = &text[value.offset..value.offset + value.size];
//                println!("nonword: \"{nonword}\"");
//                espeak_phonemes.push_str(nonword);
//                phonetisaurus_phonemes.push_str(nonword);
//                combined_phonemes.push_str(nonword);
//            },
//            WordOrNonWord::Word(value) => {
//                let word = text[value.offset..value.offset + value.size].trim();
//                println!("word:    \"{word}\"");
//                let espeak_word_phonemes = {
//                    let phonemes = _text_to_phonemes(word, "en-us", None, true, false).unwrap().join("");
//                    let mut phonemes = phonemes.trim().to_string();
//                    if phonemes.ends_with(".") && !word.ends_with(".") {
//                        phonemes.pop();
//                    }
//                    espeak_phonemes.push_str(&phonemes);
//                    phonemes
//                    //println!("\tespeak:          \"{phonemes}\"");
//                };
//                let phonetisaurus_word_phonemes = match self.phonetisaurus_phonemizer.phonemize_word(word) {
//                    Ok(phonemes_result) => {
//                        println!("\tphonetisaurus:    \"{}\"", &phonemes_result.phonemes);
//                        phonetisaurus_phonemes.push_str(&phonemes_result.phonemes);
//                        Some(phonemes_result.phonemes)
//                    }
//                    Err(err) => {
//                        println!("\tphonetisaurus error:    \"{err}\"");
//                        None
//                    }
//                };
//                match phonetisaurus_word_phonemes {
//                    Some(phonemes) => {
//                        combined_phonemes.push_str(&phonemes);
//                    }
//                    None => {
//                        combined_phonemes.push_str(&espeak_word_phonemes);
//                    }
//                }
//            }
//        }
//    }
//    println!("espeak_phonemes:        \"{espeak_phonemes}\"");
//    println!("phonetisaurus_phonemes: \"{phonetisaurus_phonemes}\"");
//    println!("combined_phonemes:      \"{combined_phonemes}\"");
//    if use_espeak {
//        return espeak_phonemes;
//    } else {
//        return phonetisaurus_phonemes;
//    }
//    //return combined_phonemes;
//}
}


/// Main TTS engine struct
pub struct TtsEngine {
    sessions: Vec<Arc<Mutex<Session>>>,
    voices: HashMap<String, Vec<Vec<f32>>>,
    phonemizer: Phonemizer
}

impl TtsEngine {
    /// Create a new TTS engine, downloading model files if necessary
    /// Uses ~/.cache/kokoros for shared model storage
    pub async fn new() -> Result<Self, String> {
        let cache_dir = get_cache_dir();
        let model_path = cache_dir.join("kokoro-v1.0.onnx");
        let voices_path = cache_dir.join("voices-v1.0.bin");

        Self::with_paths(
            model_path.to_str().unwrap_or("kokoro-v1.0.onnx"),
            voices_path.to_str().unwrap_or("voices-v1.0.bin")
        ).await
    }

    fn create_sessions(model_path: &str) -> Vec<Arc<Mutex<Session>>> {
        // Load ONNX model
        let model_bytes = std::fs::read(model_path)
            .map_err(|e| format!("Failed to read model file: {}", e)).unwrap();
        
        let mut sessions = Vec::new();

        if let Ok(session) = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e)).unwrap()
            .with_execution_providers([ROCmExecutionProvider::default().build()])
            .map_err(|e| format!("Failed to set rocm: {}", e)).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {}", e)).unwrap()
            .with_intra_threads(std::thread::available_parallelism().unwrap().get() - 1)
            .map_err(|e| format!("Failed to set intra threads: {}", e)).unwrap()
            .commit_from_memory(&model_bytes)
            .map_err(|e| format!("Failed to load model: {}", e)) {
                sessions.push(Arc::new(Mutex::new(session)));
            }

        if let Ok(session) = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e)).unwrap()
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .map_err(|e| format!("Failed to set cuda: {}", e)).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {}", e)).unwrap()
            .with_intra_threads(std::thread::available_parallelism().unwrap().get() - 1)
            .map_err(|e| format!("Failed to set intra threads: {}", e)).unwrap()
            .commit_from_memory(&model_bytes)
            .map_err(|e| format!("Failed to load model: {}", e)) {
                sessions.push(Arc::new(Mutex::new(session)));
            }

        if let Ok(session) = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e)).unwrap()
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| format!("Failed to set cuda: {}", e)).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {}", e)).unwrap()
            .with_intra_threads(std::thread::available_parallelism().unwrap().get() - 1)
            .map_err(|e| format!("Failed to set intra threads: {}", e)).unwrap()
            .commit_from_memory(&model_bytes)
            .map_err(|e| format!("Failed to load model: {}", e)) {
                sessions.push(Arc::new(Mutex::new(session)));
            }


        return sessions;
    }

    /// Create a new TTS engine with custom model paths
    pub async fn with_paths(model_path: &str, voices_path: &str) -> Result<Self, String> {
        // Ensure cache directory exists
        if let Some(parent) = Path::new(model_path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create cache directory: {}", e))?;
        }

        // Download model if needed
        if !Path::new(model_path).exists() {
            download_file(MODEL_URL, model_path).await
                .map_err(|e| format!("Failed to download model: {}", e))?;
        }

        // Download voices if needed
        if !Path::new(voices_path).exists() {
            download_file(VOICES_URL, voices_path).await
                .map_err(|e| format!("Failed to download voices: {}", e))?;
        }

        let sessions = Self::create_sessions(&model_path);

        // Load voices
        let voices = load_voices(voices_path)
            .map_err(|e| format!("Failed to load voices: {}", e))?;

        Ok(Self {
            sessions,
            voices,
            phonemizer: Phonemizer::new()
        })
    }

    /// Synthesize speech from text
    pub fn synthesize_async(&mut self, texts: &Vec<&str>, voice: Option<&str>) -> Vec<Vec<f32>> {

        let voice = voice.unwrap_or(DEFAULT_VOICE);

        let mut audios = Vec::new();

        for text in texts {
            let mut audio: Vec<f32> = Vec::new();

            //let replaced_clauses = text.replace(";", "\n");

            let mut sentences: VecDeque<(&str, VecDeque<Vec<i64>>)> = text
                .lines()
                .map(|f| f.unicode_sentences())
                .flatten()
                .map(|t| (t, self.phonemizer.graphemes_to_phonemes(t, true)))
                .collect();

            let mut to_infer: Vec<i64> = Vec::new();


            while sentences.len() > 0 && sentences[0].1.len() > 0 {
                if (to_infer.len() + sentences[0].1[0].len()) < 509 {
                    to_infer.extend(sentences[0].1.pop_front().unwrap());
                    if sentences[0].1.len() == 0 {
                        sentences.pop_front();
                    }

                    continue;
                }

                if to_infer.len() == 0 {
                    panic!("Whoopsies, don't know how to split a really long sentence ({} tokens): {}", sentences[0].1[0].len(), &sentences[0].0);
                } else {
                    //audio.extend(self.infer(vec![to_infer], voice, DEFAULT_SPEED).unwrap());
                    to_infer = Vec::new();
                }
            }

            //audios.push(audio);
        }

        return audios;
    }

//    pub fn synthesize(&mut self, text: &str, voice: Option<&str>) -> Result<Vec<f32>, String> {
//        let voice = voice.unwrap_or(DEFAULT_VOICE);
//
//        let mut audio = Vec::new();
//
//        //let phonemes_vec = vec!["$ɡˈɪmi ɐ bɹˈeɪk!$", "$aɪ wˈʌzn̩t tɹˈaɪɪŋ təbi fˈʌni. ænd kˈʌm ˈɔn. ðæt sˈʌmtaɪmz wʌz ʌnnˈɛsᵻsɚɹi.$"];
//
//        for (i, sentence) in text.unicode_sentences().enumerate() {
//            //println!("{i}: \"{sentence}\"");
//
//            // Convert text to phonemes
//            // Parameters: text, language, voice variant (None for default), preserve punctuation, with_stress
//
//            //let phonemes = text_to_phonemes(sentence.trim(), "en-us", None, true, false)
//            //    .map_err(|e| format!("Failed to convert text to phonemes: {:?}", e))?;
//
//            // Join phonemes into a single string
//            let mut phonemes_str = self.phonemizer.graphemes_to_phonemes(sentence, true);
//            phonemes_str.insert(0, '$');
//            phonemes_str.push('$');
//            
//            // Tokenize phonemes using proper vocabulary
//            let tokens = self.tokenize_phonemes(&phonemes_str);
//
//            //let tokens = self.tokenize(&phonemes_vec[i]);
//            
//
//            // Run inference
//            // TODO: I should rearchitect this so I don't have to clone the style, and remake the style and
//            //       speed tensors within infer every call.
//            //let self.sessions.pop()
//            //let session = self.sessions[0].lock().unwrap();
//            //audio.extend(self.infer(session, tokens, voice, DEFAULT_SPEED)?);
//            audio.extend(self.infer(tokens, voice, DEFAULT_SPEED)?);
//        }
//
//        return Ok(audio);

//        // Convert text to phonemes
//        // Parameters: text, language, voice variant (None for default), preserve punctuation, with_stress
//        let phonemes = text_to_phonemes(text, "en", None, true, true)
//            .map_err(|e| format!("Failed to convert text to phonemes: {:?}", e))?;
//
//        // Join phonemes into a single string
//        let mut phonemes_str = phonemes.join("");
//        phonemes_str.insert(0, '$');
//        phonemes_str.push('$');
//        
//        println!("phonemes [{}]: {}", phonemes.len(), phonemes_str);
//
//        // Tokenize phonemes using proper vocabulary
//        let tokens = self.tokenize(&phonemes_str);
//
//        // Run inference
//        let audio = self.infer(tokens, style, DEFAULT_SPEED)?;
//
//
//        Ok(audio)
//    }

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
        let source = Decoder::new(cursor)
            .map_err(|e| format!("Failed to decode audio: {}", e))?;

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
        use cpal::traits::{HostTrait, DeviceTrait};

        if let Ok(devices) = cpal::default_host().output_devices() {
            devices.filter_map(|device| device.name().ok()).collect()
        } else {
            vec!["default".to_string()]
        }
    }

    /// Convert audio to WAV bytes (for playback)
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
                writer.write_sample(sample)
                    .map_err(|e| format!("Failed to write sample: {}", e))?;
            }

            writer.finalize()
                .map_err(|e| format!("Failed to finalize WAV: {}", e))?;
        }

        Ok(buffer)
    }

    /// List available voices
    pub fn voices(&self) -> Vec<String> {
        self.voices.keys().cloned().collect()
    }

    /// Save audio to WAV file
    pub fn save_wav(&self, path: &str, audio: &[f32]) -> Result<(), String> {
        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
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
            writer.write_sample(sample)
                .map_err(|e| format!("Failed to write sample: {}", e))?;
        }

        writer.finalize()
            .map_err(|e| format!("Failed to finalize WAV: {}", e))?;
        Ok(())
    }

    #[cfg(feature = "mp3")]
    /// Save audio to MP3 file (requires mp3 feature)
    pub fn save_mp3(&self, path: &str, audio: &[f32]) -> Result<(), String> {
        use mp3lame_encoder::{Builder, Encoder, FlushNoGap};

        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Convert to i16 samples
        let samples: Vec<i16> = audio.iter()
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
        let encoded = encoder.encode(&samples)
            .map_err(|e| format!("Failed to encode: {:?}", e))?;
        mp3_data.extend_from_slice(&encoded);

        let encoded = encoder.flush::<FlushNoGap>()
            .map_err(|e| format!("Failed to flush: {:?}", e))?;
        mp3_data.extend_from_slice(&encoded);

        // Write to file
        let mut file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        file.write_all(&mp3_data)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    #[cfg(feature = "opus-format")]
    /// Save audio to OPUS file - great for streaming and low bandwidth!
    pub fn save_opus(&self, path: &str, audio: &[f32], bitrate: i32) -> Result<(), String> {
        use audiopus::{coder::Encoder, Channels, SampleRate, Application};

        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Convert to i16 samples
        let samples: Vec<i16> = audio.iter()
            .map(|&s| (s * 32767.0).clamp(-32768.0, 32767.0) as i16)
            .collect();

        // Setup OPUS encoder (24kHz mono)
        let mut encoder = Encoder::new(
            SampleRate::Hz24000,
            Channels::Mono,
            Application::Audio
        ).map_err(|e| format!("Failed to create OPUS encoder: {:?}", e))?;

        // Set bitrate (typical: 24000 for speech)
        encoder.set_bitrate(bitrate)
            .map_err(|e| format!("Failed to set bitrate: {:?}", e))?;

        // Encode in chunks (OPUS needs specific frame sizes)
        let frame_size = 480; // 20ms at 24kHz
        let mut opus_data = Vec::new();

        for chunk in samples.chunks(frame_size) {
            let mut encoded = vec![0u8; 4000];
            let len = encoder.encode(chunk, &mut encoded)
                .map_err(|e| format!("Failed to encode OPUS: {:?}", e))?;
            opus_data.extend_from_slice(&encoded[..len]);
        }

        // Write to file
        let mut file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;
        file.write_all(&opus_data)
            .map_err(|e| format!("Failed to write file: {}", e))?;

        Ok(())
    }

    #[cfg(feature = "flac-format")]
    /// Save audio to FLAC file - lossless quality for when you're getting FLAC!
    pub fn save_flac(&self, path: &str, audio: &[f32]) -> Result<(), String> {
        use flac::StreamWriter;

        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        // Convert to i32 samples (24-bit audio in 32-bit container)
        let samples: Vec<i32> = audio.iter()
            .map(|&s| (s * 8388607.0).clamp(-8388608.0, 8388607.0) as i32)
            .collect();

        // Create FLAC writer
        let file = File::create(path)
            .map_err(|e| format!("Failed to create file: {}", e))?;

        let mut writer = StreamWriter::new(file, 24)
            .map_err(|e| format!("Failed to create FLAC writer: {:?}", e))?;

        // Write samples
        for sample in samples {
            writer.write_sample(sample)
                .map_err(|e| format!("Failed to write FLAC sample: {:?}", e))?;
        }

        // Finalize
        writer.finalize()
            .map_err(|e| format!("Failed to finalize FLAC: {:?}", e))?;

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
            #[cfg(feature = "flac-format")]
            "flac" => self.save_flac(path, audio),
            _ => Err(format!("Unsupported audio format: {}", extension))
        }
    }

    // Internal methods

    //fn infer(&mut self, session: std::sync::MutexGuard<'_, Session>, tokens: Vec<Vec<i64>>, voice: &str, speed: f32) -> Result<Vec<f32>, String> {
    fn infer(&mut self, tokens: Vec<Vec<i64>>, voice: &str, speed: f32) -> Result<Vec<f32>, String> {
        //let mut session = session;
        let mut session = self.sessions[0].lock().unwrap();

        // Prepare tokens tensor
        let tokens_shape = [tokens.len(), tokens[0].len()];
        let tokens_flat: Vec<i64> = tokens.into_iter().flatten().collect();
        let num_tokens = tokens_flat.len();
        println!("tokens: {:?}", &tokens_flat);

        let tokens_tensor = Tensor::from_array((tokens_shape, tokens_flat))
            .map_err(|e| format!("Failed to create tokens tensor: {}", e))?;

        // Prepare style tensor
        let voice = self.voices.get(voice)
            .ok_or_else(|| format!("Voice '{}' not found", voice))?;

        let style = voice[num_tokens].clone();
        let style_shape = [1, style.len()];
        let style_tensor = Tensor::from_array((style_shape, style))
            .map_err(|e| format!("Failed to create style tensor: {}", e))?;

        // Prepare speed tensor
        let speed_tensor = Tensor::from_array(([1], vec![speed]))
            .map_err(|e| format!("Failed to create speed tensor: {}", e))?;

        // Create inputs
        use std::borrow::Cow;
        let inputs = SessionInputs::from(vec![
            (Cow::Borrowed("tokens"), SessionInputValue::Owned(Value::from(tokens_tensor))),
            (Cow::Borrowed("style"), SessionInputValue::Owned(Value::from(style_tensor))),
            (Cow::Borrowed("speed"), SessionInputValue::Owned(Value::from(speed_tensor))),
        ]);

        // Run inference
        let outputs = session.run(inputs)
            .map_err(|e| format!("Failed to run inference: {}", e))?;

        // Extract audio
        let (_shape, data) = outputs["audio"]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("Failed to extract audio tensor: {}", e))?;

        Ok(data.to_vec())
    }
}

// Helper functions

// Build proper vocabulary for tokenization (matching original Kokoros)
fn build_vocab() -> HashMap<char, i64> {
    let pad = "$";
    let punctuation = r#";:,.!?¡¿—…"«»"" "#;
    let letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    let letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ";

    let symbols: String = [pad, punctuation, letters, letters_ipa].concat();

    let ret: HashMap<char, i64> = symbols
        .chars()
        .enumerate()
        .map(|(idx, c)| (c, idx as i64))
        .collect();

//    let mut temp: Vec<(char, i64)> = ret
//        .iter()
//        .map(|(character, index)| (*character, *index))
//        .collect();
//
//    temp.sort_by_key(|(_character, index)| *index);
//
//    for (character, index) in temp {
//        println!("\t\"{}\":{}", character, index);
//    }

    // let ret: HashMap<char, i64> = HashMap::from([
    //     (';', 1),
    //     (':', 2),
    //     (',', 3),
    //     ('.', 4),
    //     ('!', 5),
    //     ('?', 6),
    //     ('—', 9),
    //     ('…', 10),
    //     ('\"', 11),
    //     ('(', 12),
    //     (')', 13),
    //     ('“', 14),
    //     ('”', 15),
    //     (' ', 16),
    //     ('\u{0303}', 17),
    //     ('ʣ', 18),
    //     ('ʥ', 19),
    //     ('ʦ', 20),
    //     ('ʨ', 21),
    //     ('ᵝ', 22),
    //     ('\u{AB67}', 23),
    //     ('A', 24),
    //     ('I', 25),
    //     ('O', 31),
    //     ('Q', 33),
    //     ('S', 35),
    //     ('T', 36),
    //     ('W', 39),
    //     ('Y', 41),
    //     ('ᵊ', 42),
    //     ('a', 43),
    //     ('b', 44),
    //     ('c', 45),
    //     ('d', 46),
    //     ('e', 47),
    //     ('f', 48),
    //     ('h', 50),
    //     ('i', 51),
    //     ('j', 52),
    //     ('k', 53),
    //     ('l', 54),
    //     ('m', 55),
    //     ('n', 56),
    //     ('o', 57),
    //     ('p', 58),
    //     ('q', 59),
    //     ('r', 60),
    //     ('s', 61),
    //     ('t', 62),
    //     ('u', 63),
    //     ('v', 64),
    //     ('w', 65),
    //     ('x', 66),
    //     ('y', 67),
    //     ('z', 68),
    //     ('ɑ', 69),
    //     ('ɐ', 70),
    //     ('ɒ', 71),
    //     ('æ', 72),
    //     ('β', 75),
    //     ('ɔ', 76),
    //     ('ɕ', 77),
    //     ('ç', 78),
    //     ('ɖ', 80),
    //     ('ð', 81),
    //     ('ʤ', 82),
    //     ('ə', 83),
    //     ('ɚ', 85),
    //     ('ɛ', 86),
    //     ('ɜ', 87),
    //     ('ɟ', 90),
    //     ('ɡ', 92),
    //     ('ɥ', 99),
    //     ('ɨ', 101),
    //     ('ɪ', 102),
    //     ('ʝ', 103),
    //     ('ɯ', 110),
    //     ('ɰ', 111),
    //     ('ŋ', 112),
    //     ('ɳ', 113),
    //     ('ɲ', 114),
    //     ('ɴ', 115),
    //     ('ø', 116),
    //     ('ɸ', 118),
    //     ('θ', 119),
    //     ('œ', 120),
    //     ('ɹ', 123),
    //     ('ɾ', 125),
    //     ('ɻ', 126),
    //     ('ʁ', 128),
    //     ('ɽ', 129),
    //     ('ʂ', 130),
    //     ('ʃ', 131),
    //     ('ʈ', 132),
    //     ('ʧ', 133),
    //     ('ʊ', 135),
    //     ('ʋ', 136),
    //     ('ʌ', 138),
    //     ('ɣ', 139),
    //     ('ɤ', 140),
    //     ('χ', 142),
    //     ('ʎ', 143),
    //     ('ʒ', 147),
    //     ('ʔ', 148),
    //     ('ˈ', 156),
    //     ('ˌ', 157),
    //     ('ː', 158),
    //     ('ʰ', 162),
    //     ('ʲ', 164),
    //     ('↓', 169),
    //     ('→', 171),
    //     ('↗', 172),
    //     ('↘', 173),
    //     ('ᵻ', 177)
    // ]);

    ret
}

fn load_voices(path: &str) -> Result<HashMap<String, Vec<Vec<f32>>>, String> {
    let mut npz = NpzReader::new(File::open(path).map_err(|e| format!("Failed to open voices file: {}", e))?)
        .map_err(|e| format!("Failed to read NPZ: {:?}", e))?;
    let mut voices: HashMap<String, Vec<Vec<f32>>> = HashMap::new();

    for name in npz.names().map_err(|e| format!("Failed to get NPZ names: {:?}", e))? {
        // Read the array directly to avoid type issues
        let arr: ArrayBase<OwnedRepr<f32>, IxDyn> = npz.by_name(&name)
            .map_err(|e| format!("Failed to read voice {}: {:?}", name, e))?;

        voices.insert(name.trim_end_matches(".npy").to_string(), Vec::new());
        let styles = voices.get_mut(&name.trim_end_matches(".npy").to_string()).unwrap();

        for i in 0..510 {
            let data = arr.as_slice()
                .ok_or_else(|| format!("Failed to get slice for voice {}", name))?[(i * 256)..(((i + 1) * 256))]
                .to_vec();
            styles.push(data);
        }
    }

    Ok(voices)
}

async fn download_file(url: &str, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    // Ensure directory exists
    if let Some(parent) = Path::new(path).parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Downloading {} to {}...", url, path);

    let response = reqwest::get(url).await?;
    let bytes = response.bytes().await?;

    let mut file = File::create(path)?;
    file.write_all(&bytes)?;

    println!("Downloaded successfully!");
    Ok(())
}

// Simple builder pattern for customization
pub struct TtsBuilder {
    model_path: String,
    voices_path: String,
}

impl Default for TtsBuilder {
    fn default() -> Self {
        let cache_dir = get_cache_dir();
        Self {
            model_path: cache_dir.join("kokoro-v1.0.onnx").to_str().unwrap_or("kokoro-v1.0.onnx").to_string(),
            voices_path: cache_dir.join("voices-v1.0.bin").to_str().unwrap_or("voices-v1.0.bin").to_string(),
        }
    }
}

impl TtsBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_path(mut self, path: &str) -> Self {
        self.model_path = path.to_string();
        self
    }

    pub fn voices_path(mut self, path: &str) -> Self {
        self.voices_path = path.to_string();
        self
    }

    pub async fn build(self) -> Result<TtsEngine, String> {
        TtsEngine::with_paths(&self.model_path, &self.voices_path).await
    }
}

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
        let _builder = TtsBuilder::new()
            .model_path("custom_model.onnx")
            .voices_path("custom_voices.bin");
    }
}