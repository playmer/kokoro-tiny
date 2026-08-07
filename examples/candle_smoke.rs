use kokoro_tiny::TtsEngine;

fn main() -> Result<(), String> {
    let model_dir = std::env::var("KOKORO_MODEL_DIR")
        .map_err(|_| "KOKORO_MODEL_DIR must point to converted Candle weights".to_string())?;
    let voice = std::env::var("KOKORO_VOICE").unwrap_or_else(|_| "af_heart".to_string());
    let voice_path = std::path::Path::new(&model_dir)
        .join("voices")
        .join(format!("{voice}.safetensors"));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .map_err(|error| error.to_string())?;
    let mut engine = runtime.block_on(TtsEngine::with_paths(
        &model_dir,
        &voice_path.to_string_lossy(),
    ))?;
    let sections = vec![(
        "Smoke test".to_string(),
        "Candle Kokoro is synthesizing this text without ONNX Runtime.\n\
         The caller still controls chunking and preserves output order.\n\
         A CUDA model and a CPU model can consume the same work queue.\n\
         Completed chunks flow directly into the audiobook encoder."
            .to_string(),
    )];
    let (audio, sample_sizes, chapters) = engine.synthesize(&sections);
    if audio.is_empty() || sample_sizes.is_empty() || chapters.len() != 1 {
        return Err("Synthesis produced incomplete output".to_string());
    }
    println!(
        "encoded_bytes={} aac_frames={} chapters={}",
        audio.len(),
        sample_sizes.len(),
        chapters.len()
    );
    Ok(())
}
