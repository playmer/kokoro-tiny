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
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use std::time::Duration;

use audio_sample::ConvertTo;
use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};
use unicode_segmentation::UnicodeSegmentation;
use espeak_rs::_text_to_phonemes;
use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ndarray_npy::NpzReader;
use ort::{
    session::{builder::GraphOptimizationLevel, Session, SessionInputs, SessionInputValue},
    value::{Tensor, Value},
};


#[cfg(feature = "playback")]
use rodio::{Decoder, OutputStream, Sink};
use std::io::Cursor;

// Constants
const MODEL_URL: &str = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx";
const VOICES_URL: &str = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin";
const SAMPLE_RATE: u32 = 24000;
//const DEFAULT_VOICE: &str = "af_sky";
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

/*
struct OffsetAndSize {
    offset: usize,
    size: usize
}

enum WordOrNonWord {
    Word(OffsetAndSize),
    NonWord(OffsetAndSize),
}

*/

impl Phonemizer {
    /*
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
     */


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
            phonemes[i] = format!("{}{}", split[0], c);
            phonemes.insert(i + 1, split[1].to_string());
            return true;
        }

        return false;
    }

    
    pub fn graphemes_to_phonemes<'a>(&self, text: &'a str, _use_espeak: bool) -> Option<(&'a str, VecDeque<Vec<i64>>)>  {
        if text.contains("It hurts it hurts it hurts. It hurts it hurts it hurts") {
            println!("Whoops.")
        }


        let mut phonemes: Vec<String> = text
            .unicode_sentences()
            .map(|s| _text_to_phonemes(s.trim(), "en-us", None, true, false)
            .unwrap()
            .join(""))
            .collect();

        if phonemes.len() == 0 {
            println!("Warning: Empty Text: {text}");
            return None
        }

        // First a combining step.
        if phonemes.len() > 1
        {
            let mut sentences = phonemes.len();
            let mut i = 0;

            while i < (sentences - 1) {
                if (phonemes[i].len() + phonemes[i + 1].len()) <= 509 {
                    let next = phonemes[i + 1].to_owned();
                    phonemes[i] += &next;
                    phonemes.remove(i + 1);

                    sentences -= 1;
                }
            
                i += 1;
            }
        }

        // Now a splitting step.
        
        let mut i = 0;
        while i != phonemes.len() {
            let mut has_stepped = false;
            while (phonemes[i].len() > 508) && (i < phonemes.len()) {
                has_stepped = false;
                if Self::split_index(&mut phonemes, i, ';') && phonemes[i].len() > 508 {
                    if phonemes[i + 1].trim().len() == 0 {
                        phonemes.pop();
                    } else {
                        continue;
                    }
                }
            
                if phonemes[i].len() > 508 && Self::split_index(&mut phonemes, i, ',') && phonemes[i].len() > 508 {
                    if phonemes[i + 1].trim().len() == 0 {
                        phonemes.pop();
                    } else {
                        continue;
                    }
                }

                if phonemes[i].len() > 508 && Self::split_index(&mut phonemes, i, '-') && phonemes[i].len() > 508 {
                    if phonemes[i + 1].trim().len() == 0 {
                        phonemes.pop();
                    } else {
                        continue;
                    }
                }

                if phonemes[i].len() > 508 && Self::split_index(&mut phonemes, i, ' ') && phonemes[i].len() > 508 {
                    if phonemes[i + 1].trim().len() == 0 {
                        phonemes.pop();
                    } else {
                        continue;
                    }
                }

                i += 1;
                has_stepped = true;
            }

            if !has_stepped {
                i += 1;
            }
        }

        let tokens: Vec<_>  = phonemes.iter().map(|f| self.tokenize_phonemes(f)).collect(); 

        let mut tokens_too_big = false;

        for token_stream in &tokens {
            if token_stream.len() > 510{
                tokens_too_big = true;
            }
        }

        if tokens_too_big {
            panic!("Detected a token string too large to synthesize: {}", text);
        }

        Some((text, tokens.into()))
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


struct SessionHandler {
    session: Arc<Mutex<Session>>,
    voice_styles: Vec<Vec<f32>>
}

impl SessionHandler {
    fn new(session: Arc<Mutex<Session>>, voice_styles: Vec<Vec<f32>>) -> SessionHandler {
        SessionHandler {
            session,
            voice_styles,
        }
    }
    
    pub fn inference(&mut self, tokens: Vec<Vec<i64>>, speed: f32) -> Result<Vec<f32>, String> {
        //let mut session = session;
        let mut session = self.session.lock().unwrap();

        // Prepare tokens tensor
        let tokens_shape = [tokens.len(), tokens[0].len()];
        let tokens_flat: Vec<i64> = tokens.into_iter().flatten().collect();
        let num_tokens = tokens_flat.len() - 2;
        //println!("tokens: {:?}", &tokens_flat);

        let tokens_tensor = Tensor::from_array((tokens_shape, tokens_flat))
            .map_err(|e| format!("Failed to create tokens tensor: {}", e))?;

        let style = self.voice_styles[num_tokens].clone();
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

struct Chunk {
    data: Vec<f32>
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
    chunks: Vec<Chunk>
}

impl Paragraph {
    fn new() -> Paragraph {
        Paragraph { finished_chunks: 0, chunks: Vec::new() }
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
                    } else if i == self.chunks.len() {
                        TrimSection::Begin
                    } else {
                        TrimSection::BeginAndEnd
                    };
                    
                    data.extend(trim_with_auto_ref(&chunk.data, trim_type, 60.0, 2048, 512).unwrap());
                }
            }
        }

        data
    }
}

struct Section {
    finished_paragraphs: usize,
    paragraphs: Vec<Paragraph> // These can be cleanly joined.
}

impl Section {
    fn new() -> Section {
        Section { finished_paragraphs: 0, paragraphs: Vec::new() }
    }
}

pub trait Encoder {
    fn feed_audio(&mut self, paragraph: Paragraph, finished: bool);
}

pub struct AACEncoder {
    remaining_samples: Vec<i16>,
    encoder: fdk_aac::enc::Encoder,
    encoder_info: fdk_aac::enc::InfoStruct,
    temp_output: Vec<u8>,
    final_output: Vec<u8>
}

pub type AACEncoderBitrate = fdk_aac::enc::BitRate;

impl AACEncoder {
    pub fn new(bit_rate: AACEncoderBitrate) -> AACEncoder {
        let params = fdk_aac::enc::EncoderParams{
            bit_rate: bit_rate,
            //bit_rate: fdk_aac::enc::BitRate::Cbr(24000),
            sample_rate: SAMPLE_RATE,
            transport: fdk_aac::enc::Transport::Raw,
            //transport: fdk_aac::enc::Transport::Adts,
            channels: fdk_aac::enc::ChannelMode::Mono,
            audio_object_type: fdk_aac::enc::AudioObjectType::Mpeg4LowComplexity
        };

        let encoder = fdk_aac::enc::Encoder::new(params).unwrap();
        let encoder_info: fdk_aac::enc::InfoStruct = encoder.info().unwrap();

        AACEncoder { 
            remaining_samples: Vec::new(),
            encoder,
            encoder_info,
            temp_output: vec![0; (6144 / 8) * 1 /* channels */],
            final_output: Vec::new()
        }
    }
}

impl Encoder for AACEncoder {
    fn feed_audio(&mut self, paragraph: Paragraph, finished: bool) {
        let coverted_audio: Vec<i16> = paragraph.combine_chunks().into_iter().map(|s| s.convert_to()).collect();
        self.remaining_samples.extend(coverted_audio);

        if !finished && self.remaining_samples.len() < (self.encoder_info.frameLength as usize) {
            return;
        }

        let frames_to_send = (self.encoder_info.frameLength as usize).min(self.remaining_samples.len());
        
        let encoding_info = self.encoder.encode(
            &self.remaining_samples[0..frames_to_send], 
            &mut self.temp_output)
            .unwrap();

        self.final_output.extend(&self.temp_output[0..encoding_info.output_size]);
    }
}



/// Main TTS engine struct
pub struct TtsEngine {
    sessions: Vec<SessionHandler>,
    phonemizer: Phonemizer
}

impl TtsEngine {
    /// Create a new TTS engine, downloading model files if necessary
    /// Uses ~/.cache/kokoros for shared model storage
    pub async fn new(voice: &str) -> Result<Self, String> {
        let cache_dir = get_cache_dir();
        let model_path = cache_dir.join("kokoro-v1.0.onnx");
        let voices_path = cache_dir.join("voices-v1.0.bin");

        Self::with_paths(
            model_path.to_str().unwrap_or("kokoro-v1.0.onnx"),
            voices_path.to_str().unwrap_or("voices-v1.0.bin"),
            voice
        ).await
    }

    fn create_sessions(model_path: &str, voices_path: &str, voice: &str) -> Vec<SessionHandler> {
        // Load voices
        let voices = load_voices(voices_path)
            .map_err(|e| format!("Failed to load voices: {}", e)).unwrap();

        let voice = &voices[voice];

        // Load ONNX model
        let model_bytes = std::fs::read(model_path)
            .map_err(|e| format!("Failed to read model file: {}", e)).unwrap();
        
        let mut sessions = Vec::new();

        if let Ok(session) = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e)).unwrap()
            .with_execution_providers([CUDAExecutionProvider::default().build().error_on_failure()])
            .map_err(|e| format!("Failed to set cuda: {}", e)).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {}", e)).unwrap()
            //.with_profiling("cuda_profiling.json")
            //.map_err(|e| format!("Failed to set profiling file: {}", e)).unwrap()
            .commit_from_memory(&model_bytes)
            .map_err(|e| format!("Failed to load model: {}", e)) {
                sessions.push(SessionHandler::new(Arc::new(Mutex::new(session)), voice.clone()));
            }

        // if let Ok(session) = Session::builder()
        //     .map_err(|e| format!("Failed to create session builder: {}", e)).unwrap()
        //     .with_execution_providers([ROCmExecutionProvider::default().build()])
        //     .map_err(|e| format!("Failed to set rocm: {}", e)).unwrap()
        //     .with_optimization_level(GraphOptimizationLevel::Level3)
        //     .map_err(|e| format!("Failed to set optimization level: {}", e)).unwrap()
        //     .with_intra_threads(std::thread::available_parallelism().unwrap().get() - 1)
        //     .map_err(|e| format!("Failed to set intra threads: {}", e)).unwrap()
        //     .commit_from_memory(&model_bytes)
        //     .map_err(|e| format!("Failed to load model: {}", e)) {
        //         sessions.push(SessionHandler::new(Arc::new(Mutex::new(session)), voice.clone()));
        //     }

        if let Ok(session) = Session::builder()
            .map_err(|e| format!("Failed to create session builder: {}", e)).unwrap()
            .with_execution_providers([CPUExecutionProvider::default().build()])
            .map_err(|e| format!("Failed to set cuda: {}", e)).unwrap()
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("Failed to set optimization level: {}", e)).unwrap()
            //.with_profiling("cpu_profiling.json")
            //.map_err(|e| format!("Failed to set profiling file: {}", e)).unwrap()
            .commit_from_memory(&model_bytes)
            .map_err(|e| format!("Failed to load model: {}", e)) {
                sessions.push(SessionHandler::new(Arc::new(Mutex::new(session)), voice.clone()));
            }


        return sessions;
    }

    /// Create a new TTS engine with custom model paths
    pub async fn with_paths(model_path: &str, voices_path: &str, voice: &str) -> Result<Self, String> {
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

        let sessions = Self::create_sessions(&model_path, voices_path, voice);

        Ok(Self {
            sessions,
            phonemizer: Phonemizer::new()
        })
    }

    fn infer_thread (
        inference_queue: Arc<lockfree::queue::Queue<(usize, usize, usize, Vec<i64>)>>,
        audios_destination: Arc<Mutex<Vec<Section>>>,
        session: SessionHandler, 
        finished_queueing: Arc<AtomicBool>) -> SessionHandler {
        let audios_destination = audios_destination;
        let mut session = session;

        let mut item: Option<(usize, usize, usize, Vec<i64>)> = inference_queue.pop();
        
        while !finished_queueing.load(std::sync::atomic::Ordering::Relaxed) {
            if let Some((i, j, k, to_infer)) = item {
                let audio = session.inference(vec![to_infer], DEFAULT_SPEED).unwrap();

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
    pub fn synthesize(&mut self, sections: &[&str]) -> Vec<Vec<f32>> {
        let inference_queue: Arc<lockfree::queue::Queue<(usize, usize, usize, Vec<i64>)>> = Arc::new(lockfree::queue::Queue::new());

        let audios_destination: Arc<Mutex<Vec<Section>>> = Arc::new(Mutex::new(Vec::new()));

        let finished_queueing: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));

        let mut handles = Vec::new();

        for _ in 0..self.sessions.len()  {
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

                let mut paragraphs: VecDeque<(&str, VecDeque<Vec<i64>>)> = section
                    .lines()
                    .filter_map(|t| self.phonemizer.graphemes_to_phonemes(t, true))
                    .collect();

                for paragraph in &mut paragraphs {
                    {
                        let mut destination = audios_destination.lock().unwrap();
                        destination[i].paragraphs.push(Paragraph::new());
                    }

                    for k in 0..paragraph.1.len(){
                        {
                            let mut destination = audios_destination.lock().unwrap();
                            destination[i].paragraphs[j].chunks.push(Chunk::new());
                        }

                        let mut chunk = paragraph.1.pop_front().unwrap();
                        chunk.insert(0, 0);
                        chunk.push(0);

                        if chunk.len() > 510 {
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
        
        let mut audios: Vec<Vec<f32>> = Vec::new();

        let mut samples_per_section: Vec<usize> = Vec::new();
        let mut temp_wave_samples: Vec<f32> = Vec::new();

        for i in 0..sections.len() {
            std::thread::sleep(Duration::from_millis(1000));
            let mut copied_audio = false;
            samples_per_section.push(0);
            let mut paragraph_index = 0;

            while !copied_audio {
                std::thread::sleep(Duration::from_millis(1000));
                let generated_audios = audios_destination.clone();

                {
                    let mut generated_audios = generated_audios.lock().unwrap();

                    let length = generated_audios.len();
                    for generated_paragraph in &mut generated_audios[i].paragraphs[paragraph_index..length] {
                        if generated_paragraph.finished_chunks != generated_paragraph.chunks.len() {
                            break;
                        }

                        paragraph_index += 1;
                        let paragraph_data = generated_paragraph.combine_chunks();
                        samples_per_section[i] += paragraph_data.len();
                        temp_wave_samples.extend(paragraph_data);

                        // Free this up now.
                        generated_paragraph.chunks.clear();
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
            std::fs::write(&format!("audio_{i}.txt"), sections[i]).unwrap();
        }
        
        finished_queueing.store(true, std::sync::atomic::Ordering::SeqCst);

        for thread in handles {
            self.sessions.push(thread.join().unwrap());
        }

        return audios;
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
            _ => Err(format!("Unsupported audio format: {}", extension))
        }
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

    //pub async fn build(self) -> Result<TtsEngine, String> {
    //    TtsEngine::with_paths(&self.model_path, &self.voices_path).await
    //}
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

pub fn amplitude_to_db(
    s: &[f32],
    ref_val: f32,
    amin: f32,
    top_db: Option<f32>,
) -> Vec<f32> {
    let magnitude: Vec<f32> = s.iter().map(|&x| x.abs()).collect();
    let power: Vec<f32> = magnitude.iter().map(|&x| x * x).collect();
    power_to_db(&power, ref_val * ref_val, amin * amin, top_db)
}

pub fn power_to_db(
    s: &[f32],
    ref_val: f32,
    amin: f32,
    top_db: Option<f32>,
) -> Vec<f32> {
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
pub fn frames_to_samples(
    frames: &[usize],
    hop_length: usize,
    n_fft: Option<usize>,
) -> Vec<usize> {
    let offset = n_fft.map_or(0, |n| n / 2);
    frames.iter().map(|&f| f * hop_length + offset).collect()
}


pub enum TrimSection {
    BeginAndEnd,
    Begin,
    End
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
        let _builder = TtsBuilder::new()
            .model_path("custom_model.onnx")
            .voices_path("custom_voices.bin");
    }
}