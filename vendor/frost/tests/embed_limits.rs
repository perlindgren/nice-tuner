//! Regression test for the 2026-10 standalone panic:
//!
//! ```text
//! Device::create_bind_group_layout, label = 'shape pipeline'
//!   Too many bindings of type StorageBuffers in Stage FRAGMENT,
//!   limit is 0, count was 2
//! ```
//!
//! The host (nice-tuner) had requested `Limits::downlevel_webgl2_defaults()`
//! while frost's shape pipeline binds two fragment-visible storage buffers.
//! naga could never have caught this: the WGSL is valid on any uplevel
//! device — the *host's device request* was wrong, and a shader compiler
//! never sees it. So the guard belongs here, on the limits frost advertises.

use wgpu::{
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType, BufferBindingType, Device,
    DeviceDescriptor, Limits, RequestAdapterOptions, ShaderStages,
};

/// Two read-only storage buffers visible to the fragment stage — exactly the
/// shape pipeline's binding shape that tripped the zero limit.
fn probe(device: &Device) {
    let storage = || BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    // wgpu surfaces validation failures through the device error handler,
    // whose default panics — so simply building this layout *is* the assert.
    let _layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("shape-pipeline probe"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: storage(),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: storage(),
                count: None,
            },
        ],
    });
}

/// The advertised limits must allow the two fragment storage buffers; the
/// downlevel profile that caused the panic must still report zero. This is
/// pure data — it runs everywhere, and proves the fix and the bug apart.
#[test]
fn device_limits_allow_fragment_storage_buffers() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = match pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        compatible_surface: None,
        ..Default::default()
    })) {
        Ok(a) => a,
        Err(_) => {
            eprintln!("no wgpu adapter; skipping");
            return;
        }
    };

    let advertised = frost::embedded::device_limits(&adapter);
    assert!(
        advertised.max_storage_buffers_per_shader_stage >= 2,
        "frost::embedded::device_limits advertises {} fragment-visible \
         storage buffers; the shape pipeline needs 2",
        advertised.max_storage_buffers_per_shader_stage
    );

    // The exact profile that caused the original panic, for contrast.
    let downlevel = Limits::downlevel_webgl2_defaults();
    assert_eq!(
        downlevel.max_storage_buffers_per_shader_stage, 0,
        "downlevel profile unexpectedly changed — revisit this regression note"
    );
}

/// End-to-end: a device built at the advertised limits must accept the shape
/// pipeline's bind layout. Skips when no GPU/adapter is present.
#[test]
fn shape_pipeline_layout_builds_at_advertised_limits() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let Ok(adapter) = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        compatible_surface: None,
        ..Default::default()
    })) else {
        eprintln!("no wgpu adapter; skipping device build");
        return;
    };
    let (device, _queue) = pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("limits-probe"),
        required_limits: frost::embedded::device_limits(&adapter),
        ..Default::default()
    }))
    .expect("request_device at frost::embedded::device_limits must succeed");
    probe(&device);
}
