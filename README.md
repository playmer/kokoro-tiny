# kokoro-tiny 🎤

[![Crates.io](https://img.shields.io/crates/v/kokoro-tiny.svg)](https://crates.io/crates/kokoro-tiny)
[![Documentation](https://docs.rs/kokoro-tiny/badge.svg)](https://docs.rs/kokoro-tiny)
[![Apache-2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Minimal, blazing-fast TTS (Text-to-Speech) crate powered by the Kokoro model (82M params). Perfect for embedding in applications, system alerts, and smart tools!

The audiobook pipeline uses multiple independent model workers that consume a shared
chunk queue while preserving source order in the encoded output.

## Features

- ⚡ **Native Candle inference** on CPU or CUDA, without ONNX Runtime
- 🧠 **Candle-native G2P** with caller-controlled chunking
- 🧵 **Concurrent model workers** configured independently for CPU and CUDA
- 🎨 **Kokoro voice packs** loaded from converted SafeTensors
- 📚 **Ordered AAC streaming** with chapter and sample-size metadata
- 💾 **Local model storage** via `KOKORO_MODEL_DIR`

## Quick Start

The model directory must contain:

```text
config.json
model.safetensors
voices/
  af_heart.safetensors
```

Set `KOKORO_MODEL_DIR` to that directory. `KOKORO_CUDA_WORKERS` and
`KOKORO_CPU_WORKERS` control the number of independently loaded models; both
default to one when CUDA is enabled.

Add to your `Cargo.toml`:

```toml
[dependencies]
kokoro-tiny = "0.1"
```

## Usage

### As a Library

```rust
use kokoro_tiny::TtsEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut tts = TtsEngine::new("af_heart").await?;
    let sections = vec![("Chapter 1".to_string(), "Hello, world!".to_string())];
    let (aac, sample_sizes, chapter_markers) = tts.synthesize(&sections);

    Ok(())
}
```

## Available Voices

50+ voices across accents and styles:

- **American**: af_sky, af_bella, am_adam, am_michael
- **British**: bf_emma, bm_george
- **Special**: af_heart (warm), am_echo (clear)

## Features

Enable additional capabilities:

```toml
[dependencies]
kokoro-tiny = { version = "0.1", features = ["all-formats"] }
```

- `playback` - Direct audio playback (default)
- `mp3` - MP3 encoding support
- `opus-format` - OPUS for streaming/VoIP
- `cuda` - GPU acceleration
- `all-formats` - All audio formats

The lower-level `candle-kokoro` crate exposes a `cudnn` feature, but this
adapter uses plain CUDA because Candle 0.10.2's thread-local cuDNN handles
cannot be safely torn down by the reusable worker lifecycle.

## Smoke Test

```bash
$env:KOKORO_MODEL_DIR = "C:\models\kokoro"
cargo run --no-default-features --features cuda --example candle_smoke
```

## Performance

- **Time-to-first-audio**: 0.5-2 seconds
- **Model size**: 82M parameters
- **Audio quality**: 24kHz sample rate
- **Memory usage**: ~200MB with model loaded

## Use Cases

Perfect for:

- 🔔 **System notifications** - Build alerts, test results
- 📊 **Smart tools** - Audio context summaries
- 🎮 **Game development** - Dynamic NPC voices
- 📱 **Accessibility** - Screen reader functionality
- 🤖 **Automation** - Voice announcements for scripts

## Model Details

kokoro-tiny uses the [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) model:
- Lightweight transformer architecture
- Trained on extensive speech datasets
- Optimized for CPU inference

## Contributing

Contributions welcome! This project is maintained by Hue & Aye at [8b.is](https://8b.is).

For the full Kokoro implementation with advanced features, check out [Kokoros](https://github.com/8b-is/Kokoros).

## License

Apache 2.0 - See [LICENSE](https://github.com/8b-is/kokoro-tiny/blob/main/LICENSE)

---

Built with 🎉 by the 8b.is team | Powered by the amazing Kokoro model