//! The render backend: the wgpu device, the surface handed over by the
//! embedding host, and the frame pipeline that turns a frame's draw list
//! into scissored draw calls.
//!
//! Everything here is internal to the crate except the embeddable engine
//! (`Canvas`, `Context`, `Process`, and the [`Embedded`](embed::Embedded)
//! frame loop, re-exported as [`frost::embedded`](crate::embedded)). Each
//! frame the backend runs the user's [`Process`] through a [`Context`],
//! updates the scene tree, draws it into a fresh draw list, expands text
//! into glyph quads, sorts the frame's [`Draw`]ings by z, and renders them
//! one scissored draw call at a time to the host's surface.

mod app;
mod bloom;
pub(crate) use app::*;

mod copy;

mod frame;
pub(crate) use frame::*;

mod windowing;

pub(crate) mod embed;
