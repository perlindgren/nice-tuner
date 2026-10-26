//! Standalone target for testing the tuner without a DAW.

use nice_plug::prelude::*;
use nice_tuner::TunerPlugin;

fn main() {
    nice_export_standalone::<TunerPlugin>();
}
