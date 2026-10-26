//! Note naming: equal-tempered note lookup for a detected frequency.

/// Note table (sharps) indexed by `midi % 12`.
const SHARP_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

/// Equal-tempered note nearest to a frequency, in scientific pitch notation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Note {
    /// Note name with sharps, e.g. `"C"`, `"C#"`, ... `"B"`.
    pub name: &'static str,
    /// Scientific pitch octave (A4 = octave 4).
    pub octave: i8,
    /// MIDI note number, A4 = 69.
    pub midi: i16,
    /// Deviation from the nearest semitone in cents, `-50.0..=50.0`.
    /// Negative means flat of the nearest note, positive sharp.
    pub cents: f64,
    /// Equal-tempered frequency of the nearest semitone at the given A4.
    pub nearest_freq: f64,
}

/// Map `freq` (Hz) to the nearest equal-tempered note assuming A4 = `a4` Hz
/// (typically 440.0).
///
/// MIDI is computed as `69 + 12 * log2(freq / a4)`, rounded to the nearest
/// integer (half away from zero, so a frequency exactly halfway between two
/// semitones yields `cents` of exactly `+/-50.0`). Non-finite or non-positive
/// `freq`/`a4` degrade gracefully to MIDI 0 (`C`, octave -1) instead of
/// producing NaN.
pub fn freq_to_note(freq: f64, a4: f64) -> Note {
    let valid = freq.is_finite() && a4.is_finite() && freq > 0.0 && a4 > 0.0;
    // midi_f is kept in f64 before rounding so `cents` survives extreme
    // inputs; clamped into a range where octave always fits in i8.
    let midi_f = if valid {
        69.0 + 12.0 * (freq / a4).log2()
    } else {
        0.0
    }
    .clamp(-1200.0, 12000.0);

    let midi = midi_f.round() as i16;
    let cents = if valid {
        // Deviation of the *frequency* from the equal-tempered semitone.
        1200.0 * (freq / semitone_freq(midi, a4)).log2()
    } else {
        0.0
    };
    // Round-tripping a frequency through log2 can yield -0.0 or +-50.0000...1
    // at the boundary; normalize signs and clamp into the documented range.
    let cents = if cents == 0.0 { 0.0 } else { cents };
    let cents = cents.clamp(-50.0, 50.0);

    let name_idx = ((midi as i32 % 12) + 12) % 12;
    let octave = (midi as i32).div_euclid(12) - 1;

    Note {
        name: SHARP_NAMES[name_idx as usize],
        octave: octave.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8,
        midi,
        cents,
        nearest_freq: semitone_freq(midi, a4),
    }
}

/// Equal-tempered frequency of MIDI note `midi` given A4 = `a4`.
#[inline]
fn semitone_freq(midi: i16, a4: f64) -> f64 {
    a4 * 2f64.powf((i32::from(midi) - 69) as f64 / 12.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_maps_to_a4() {
        let n = freq_to_note(440.0, 440.0);
        assert_eq!(n.name, "A");
        assert_eq!(n.octave, 4);
        assert_eq!(n.midi, 69);
        assert!(n.cents.abs() < 1e-6);
        assert!((n.nearest_freq - 440.0).abs() < 1e-9);
    }

    #[test]
    fn a_sharp4() {
        let n = freq_to_note(466.16, 440.0);
        assert_eq!(n.name, "A#");
        assert_eq!(n.octave, 4);
        assert_eq!(n.midi, 70);
        assert!(n.cents.abs() < 1.0);
    }

    #[test]
    fn low_e2() {
        let n = freq_to_note(82.41, 440.0);
        assert_eq!(n.name, "E");
        assert_eq!(n.octave, 2);
        assert_eq!(n.midi, 40);
    }

    #[test]
    fn boundary_is_plus_minus_50() {
        // Exactly halfway (in log domain) between A4 (69) and A#4 (70):
        // geometric mean of 440 and 466.1638...
        let boundary = 440.0 * 2f64.powf(0.5 / 12.0);
        let n = freq_to_note(boundary, 440.0);
        assert!((n.cents.abs() - 50.0).abs() < 0.6, "cents = {}", n.cents);
    }

    #[test]
    fn c0_and_c8_octaves() {
        let c0 = freq_to_note(16.351_597_83, 440.0); // MIDI 12 => C0? no: MIDI 12 = C0? C0 = MIDI 12? MIDI 12 = C0 (C-1 = 0)
        assert_eq!(c0.midi, 12);
        assert_eq!(c0.name, "C");
        assert_eq!(c0.octave, 0);
        let c8 = freq_to_note(4186.009, 440.0);
        assert_eq!(c8.midi, 108);
        assert_eq!(c8.name, "C");
        assert_eq!(c8.octave, 8);
    }

    #[test]
    fn invalid_input_degrades() {
        let n = freq_to_note(f64::NAN, 440.0);
        assert_eq!(n.midi, 0);
        assert_eq!(n.name, "C");
        assert!(n.cents.is_finite());
        let n2 = freq_to_note(-5.0, 440.0);
        assert_eq!(n2.midi, 0);
    }
}
