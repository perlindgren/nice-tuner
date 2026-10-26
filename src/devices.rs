//! cpal-backed enumeration for the standalone settings page.
//!
//! Compiled into the standalone build only: a plugin inside a host must not
//! enumerate audio systems it does not own. Everything here runs on the GUI
//! thread, on demand (page entry, backend change) — never on the audio
//! thread, and never while merely displaying the tuner.
//!
//! Caveats the UI inherits and states honestly:
//! - ASIO enumeration *loads* each registered driver to read its properties;
//!   with several drivers installed, opening the page can pause for a moment.
//! - Device lists are cpal's device names; for ASIO that is the driver name
//!   (ports are not addressable through cpal, cf. README).

use cpal::traits::{DeviceTrait, HostTrait};

/// A selectable backend: nice-plug's `BackendType` value + a display label.
#[derive(Debug, Clone)]
pub struct BackendOption {
    pub id: String,
    pub label: String,
}

fn host_id_value(id: cpal::HostId) -> Option<(&'static str, &'static str)> {
    // Only hosts nice-plug's BackendType can actually select via cpal.
    // Each variant exists only where cpal builds it (platform/feature cfg);
    // if-chains keep that exhaustiveness-free.
    // cpal builds the Asio variant only on Windows, feature or not.
    #[cfg(all(windows, feature = "asio-host"))]
    if id == cpal::HostId::Asio {
        return Some(("asio", "ASIO"));
    }
    #[cfg(windows)]
    if id == cpal::HostId::Wasapi {
        return Some(("wasapi", "Windows WASAPI"));
    }
    #[cfg(target_os = "macos")]
    if id == cpal::HostId::CoreAudio {
        // Key must match nice-plug's clap ValueEnum spelling (`core-audio`,
        // kebab-case) — a cpal-style `coreaudio` here poisons the settings
        // file and the next startup dies at argv parsing, never reaching the
        // device. Reads tolerate both spellings (see settings migration).
        return Some(("core-audio", "macOS Core Audio"));
    }
    #[cfg(target_os = "linux")]
    if id == cpal::HostId::Alsa {
        return Some(("alsa", "Linux ALSA"));
    }
    let _ = id;
    None
}

fn host_id_from_value(value: &str) -> Option<cpal::HostId> {
    #[cfg(all(windows, feature = "asio-host"))]
    if value == "asio" {
        return Some(cpal::HostId::Asio);
    }
    #[cfg(windows)]
    if value == "wasapi" {
        return Some(cpal::HostId::Wasapi);
    }
    #[cfg(target_os = "macos")]
    if value == "coreaudio" || value == "core-audio" {
        return Some(cpal::HostId::CoreAudio);
    }
    #[cfg(target_os = "linux")]
    if value == "alsa" {
        return Some(cpal::HostId::Alsa);
    }
    let _ = value;
    None
}

/// Every backend the running binary can offer, nice-plug `auto` first.
/// (ASIO appears only when compiled via `--features standalone-asio`, since
/// `available_hosts` reports compiled hosts; `jack` is nice-plug's own
/// backend, always a valid choice to name even where no server runs.)
pub fn available_backends() -> Vec<BackendOption> {
    let mk = |id: &str, label: &str| BackendOption {
        id: id.to_owned(),
        label: label.to_owned(),
    };
    let mut v = vec![mk("auto", "auto (jack if present, else system default)")];
    for h in cpal::available_hosts() {
        if let Some((id, label)) = host_id_value(h) {
            v.push(mk(id, label));
        }
    }
    v.push(mk("jack", "JACK"));
    v.push(mk("dummy", "no audio (dummy)"));
    v
}

fn open_host(backend: &str) -> Option<cpal::Host> {
    cpal::host_from_id(host_id_from_value(backend)?).ok()
}

fn device_names(devices: impl Iterator<Item = cpal::Device>) -> Vec<String> {
    let mut v: Vec<String> = devices
        .filter_map(|d| d.description().ok().map(|x| x.name().to_owned()))
        .collect();
    v.sort();
    v.dedup();
    v
}

pub fn list_input_devices(backend: &str) -> Vec<String> {
    open_host(backend)
        .and_then(|h| h.input_devices().ok())
        .map(device_names)
        .unwrap_or_default()
}

pub fn list_output_devices(backend: &str) -> Vec<String> {
    open_host(backend)
        .and_then(|h| h.output_devices().ok())
        .map(device_names)
        .unwrap_or_default()
}

/// The system-default input/output device NAMES under this backend
/// ("auto"/empty = the platform's default host). The settings page must
/// name what "(default)" actually is — a dropdown row saying the word is
/// a row saying nothing.
pub fn default_device_names(backend: &str) -> (Option<String>, Option<String>) {
    let host = if backend == "auto" || backend.is_empty() {
        cpal::default_host()
    } else {
        match open_host(backend) {
            Some(h) => h,
            None => return (None, None),
        }
    };
    let name = |d: Option<cpal::Device>| {
        d.and_then(|d| d.description().ok())
            .map(|x| x.name().to_owned())
    };
    (
        name(host.default_input_device()),
        name(host.default_output_device()),
    )
}

fn find_device(backend: &str, name: &str) -> Option<cpal::Device> {
    open_host(backend)?
        .input_devices()
        .ok()?
        .find(|d| d.description().ok().is_some_and(|x| x.name() == name))
}

/// The sample rate the DRIVER is currently set to. For ASIO this is
/// `ASIOGetSampleRate` (what the Native Instruments control panel shows),
/// for CoreAudio the device's default. The panel's "(default)" resolves
/// through here at startup because nice-plug's own default is a hard-coded
/// 48 kHz — which silently overrides a driver set to anything else.
pub fn driver_default_rate(
    backend: &str,
    input: Option<&str>,
    output: Option<&str>,
) -> Option<f32> {
    if backend == "jack" || backend == "dummy" {
        return None; // no cpal device behind these (JACK: the server decides)
    }
    let host = if backend == "auto" || backend.is_empty() {
        cpal::default_host()
    } else {
        open_host(backend)?
    };
    let in_dev = match input {
        Some(n) => host
            .input_devices()
            .ok()
            .and_then(|d| find_named(d, n))
            .or_else(|| host.default_input_device()),
        None => host.default_input_device(),
    };
    if let Some(rate) = in_dev
        .as_ref()
        .and_then(|d| d.default_input_config().ok())
        .map(|c| c.sample_rate() as f32)
    {
        return Some(rate);
    }
    let out_dev = match output {
        Some(n) => host
            .output_devices()
            .ok()
            .and_then(|d| find_named(d, n))
            .or_else(|| host.default_output_device()),
        None => host.default_output_device(),
    };
    out_dev
        .as_ref()
        .and_then(|d| d.default_output_config().ok())
        .map(|c| c.sample_rate() as f32)
}

fn find_named(mut devs: impl Iterator<Item = cpal::Device>, want: &str) -> Option<cpal::Device> {
    devs.find(|d| d.description().ok().is_some_and(|x| x.name() == want))
}

const DEFAULT_RATES: [f32; 4] = [44100.0, 48000.0, 88200.0, 96000.0];
/// Every size offered when the device permits it — the ladder reaches from
/// the 8-frame floors real ASIO control panels use up to 4096.
const BUFFER_CANDIDATES: [u32; 16] = [
    8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096,
];

/// Sample rates (Hz) and buffer sizes (frames) to offer for the given
/// backend/devices. Values come from the *real* driver: cpal queries each
/// device's supported configs (for ASIO that is `ASIOGetBufferSize`, via
/// cpal's `driver.buffersize_range()`). Buffer bounds are the INTERSECTION
/// of the input and output devices' reported ranges — a duplex session must
/// satisfy both. Unknown ranges (drivers that do not report) impose no
/// constraint; a device-less choice (e.g. `auto`) falls back to defaults.
pub fn stream_options(
    backend: &str,
    input: Option<&str>,
    output: Option<&str>,
) -> (Vec<f32>, Vec<u32>) {
    let in_dev = input.and_then(|want| find_device(backend, want));
    let out_dev = output.and_then(|want| find_device(backend, want));
    if in_dev.is_none() && out_dev.is_none() {
        return (DEFAULT_RATES.to_vec(), vec![64, 128, 256, 512, 1024]);
    }

    let in_cfgs = in_dev
        .as_ref()
        .and_then(|d| d.supported_input_configs().ok())
        .map(|c| c.collect::<Vec<_>>());
    let out_cfgs = out_dev
        .as_ref()
        .and_then(|d| d.supported_output_configs().ok())
        .map(|c| c.collect::<Vec<_>>());

    // Rates: from the input device when present (it is the one we tune with),
    // else from the output device.
    let rate_src = in_cfgs.as_ref().or(out_cfgs.as_ref());
    let mut rates: Vec<f32> = Vec::new();
    if let Some(cfgs) = rate_src {
        for c in cfgs {
            let (lo, hi) = (c.min_sample_rate(), c.max_sample_rate());
            if lo == hi {
                rates.push(lo as f32);
            } else {
                rates.extend(
                    DEFAULT_RATES
                        .into_iter()
                        .filter(|r| *r >= lo as f32 && *r <= hi as f32),
                );
            }
        }
    }
    rates.sort_by(f32::total_cmp);
    rates.dedup();
    if rates.is_empty() {
        rates = DEFAULT_RATES.to_vec();
    }

    // Buffers: intersect every reported Range; skip devices that report
    // Unknown (they will take whatever the session picks).
    let mut buf_min: Option<u32> = None;
    let mut buf_max: Option<u32> = None;
    let mut absorb = |range: &cpal::SupportedBufferSize| {
        if let cpal::SupportedBufferSize::Range { min, max } = *range {
            buf_min = Some(buf_min.map_or(min, |m| m.max(min)));
            buf_max = Some(buf_max.map_or(max, |m| m.min(max)));
        }
    };
    for c in in_cfgs.iter().flat_map(|v| v.iter()) {
        absorb(c.buffer_size());
    }
    for c in out_cfgs.iter().flat_map(|v| v.iter()) {
        absorb(c.buffer_size());
    }

    let buffers = match (buf_min, buf_max) {
        (Some(min), Some(max)) if min <= max => {
            let mut v: Vec<u32> = BUFFER_CANDIDATES
                .into_iter()
                .filter(|b| *b >= min && *b <= max)
                .collect();
            if v.is_empty() {
                v.push(min);
            }
            v
        }
        _ => vec![64, 128, 256, 512, 1024],
    };
    (rates, buffers)
}
