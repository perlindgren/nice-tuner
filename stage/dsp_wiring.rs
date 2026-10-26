// STAGING (M5): exact code to replace the two marked regions in src/lib.rs.

// ── 1. TunerPlugin struct: add field ───────────────────────────────────────
pub struct TunerPlugin {
    params: Arc<TunerParams>,
    shared: Arc<TunerSharedState>,
    detector: Option<pitch::PitchDetector>,
}
// Default impl gains: detector: None,

// ── 2. Plugin::activate: replace body ──────────────────────────────────────
fn activate(
    &mut self,
    _audio_io_layout: &AudioIOLayout,
    buffer_config: &BufferConfig,
    _context: &mut impl ActivateContext<Self>,
) -> bool {
    self.detector = Some(pitch::PitchDetector::new(pitch::Config {
        sample_rate: buffer_config.sample_rate as f64,
        min_freq: 30.0,    // 5-string bass B0
        max_freq: 2100.0,  // harmonics up past the guitar's E7 territory
        ..Default::default()
    }));
    true
}

// ── 3. Plugin::process: analysis block (buffer stays untouched = passthrough) ─
fn process(
    &mut self,
    buffer: &mut Buffer,
    _aux: &mut AuxiliaryBuffers,
    _context: &mut impl ProcessContext<Self>,
) -> ProcessStatus {
    // Param mirrors for the GUI thread (cheap, once per block).
    self.shared
        .a4_ref_bits
        .store((self.params.a4_ref.value() as f64).to_bits(), Ordering::Relaxed);
    self.shared.gate_db.store(self.params.gate_db.value());

    if let Some(detector) = self.detector.as_mut() {
        if self.shared.is_open() {
            // VERIFIED vs v0.4.2 buffer.rs: Buffer is IN-PLACE (output slices;
            // the framework already copied host input into them). `iter_blocks`
            // yields per-block `&mut [&mut [f32]]` = one contiguous slice per
            // MAIN-IO channel. Downmix = mean over channels per frame.
            const CHUNK: usize = 4096;
            let mut mono = [0f32; CHUNK];
            for mut block in buffer.iter_blocks(CHUNK) {
                let n = block.first().map(|c| c.len()).unwrap_or(0);
                let nch = block.len().max(1);
                for i in 0..n {
                    let mut acc = 0.0f32;
                    for ch in block.iter_mut() {
                        acc += ch[i];
                    }
                    mono[i] = acc / nch as f32;
                }
                if let Some(d) = detector.push_samples(&mono[..n]) {
                    self.shared.freq_hz.store(d.freq_hz);
                    self.shared.clarity.store(d.clarity);
                    self.shared.rms.store(d.rms);
                    self.shared.last_detection_ms.store(now_epoch_ms());
                }
            }
        } else {
            detector.reset();
        }
    }

    ProcessStatus::Normal
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}
