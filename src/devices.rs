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
        return Some(("coreaudio", "macOS CoreAudio"));
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
    if value == "coreaudio" {
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

fn find_device(backend: &str, name: &str) -> Option<cpal::Device> {
    open_host(backend)?
        .input_devices()
        .ok()?
        .find(|d| d.description().ok().is_some_and(|x| x.name() == name))
}

const DEFAULT_RATES: [f32; 4] = [44100.0, 48000.0, 88200.0, 96000.0];
const BUFFER_CANDIDATES: [u32; 9] = [32, 48, 64, 96, 128, 256, 512, 1024, 2048];

/// Sample rates (Hz) and buffer sizes (frames) to offer for the given
/// backend/device. Unknown or unspecified devices get sensible menus:
/// common rates and 64–1024 buffers.
pub fn stream_options(backend: &str, device: Option<&str>) -> (Vec<f32>, Vec<u32>) {
    let Some(device) = device.and_then(|want| find_device(backend, want)) else {
        return (DEFAULT_RATES.to_vec(), vec![64, 128, 256, 512, 1024]);
    };
    let configs = device
        .supported_input_configs()
        .map(|c| c.collect::<Vec<_>>())
        .unwrap_or_default();

    let mut rates: Vec<f32> = Vec::new();
    let mut buf_min = u32::MAX;
    let mut buf_max = 0u32;
    for c in &configs {
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
        if let cpal::SupportedBufferSize::Range { min, max } = c.buffer_size() {
            buf_min = buf_min.min(*min);
            buf_max = buf_max.max(*max);
        }
    }
    rates.sort_by(f32::total_cmp);
    rates.dedup();
    if rates.is_empty() {
        rates = DEFAULT_RATES.to_vec();
    }

    let mut buffers: Vec<u32> = if buf_max > 0 {
        BUFFER_CANDIDATES
            .into_iter()
            .filter(|b| *b >= buf_min && *b <= buf_max)
            .collect()
    } else {
        Vec::new()
    };
    if buffers.is_empty() {
        buffers = if buf_min <= buf_max {
            vec![buf_min]
        } else {
            vec![64, 128, 256, 512, 1024]
        };
    }
    (rates, buffers)
}
