//! Hard gate: the RT↔UI analysis handoff — the audio thread's *entire*
//! analysis contribution (downmixed block → `Producer::push_entire_slice` on the
//! preallocated rtrb ring) — performs zero heap allocations and preserves
//! the stream bit-exact. Detection itself lives on the UI side of this ring
//! (TunerUi::analyze), guarded separately by crates/pitch/tests/no_alloc.rs.
//!
//! Kept in its own test binary so the counting window has no other allocator
//! traffic: the libtest harness is parked while this runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

#[test]
fn rt_handoff_never_allocates_and_is_lossless() {
    // Same capacity as the plugin's feed (src/lib.rs TunerPlugin::default).
    let (mut producer, mut consumer) = rtrb::RingBuffer::new(1 << 15);

    // 64-frame blocks like a tight hardware callback.
    let block: Vec<f32> = (0..64).map(|i| (i as f32 * 0.1).sin()).collect();
    let mut out = [0f32; 4096];

    // Warm-up round: first touch may initialize lazily-inlined state.
    producer.push_entire_slice(&block).unwrap();
    let n = consumer.slots().min(out.len());
    consumer.pop_entire_slice(&mut out[..n]).unwrap();

    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);

    let mut pushed = 0usize;
    let mut popped = 0usize;
    for _ in 0..500 {
        // "Audio thread"
        producer.push_entire_slice(&block).unwrap();
        pushed += block.len();
        // "UI frame"
        let n = consumer.slots().min(out.len());
        consumer.pop_entire_slice(&mut out[..n]).unwrap();
        // No allocates here: compare in place against the repeating block.
        for (i, v) in out[..n].iter().enumerate() {
            assert_eq!(*v, block[i % block.len()], "sample {i} corrupted");
        }
        popped += n;
    }

    COUNTING.store(false, Ordering::Relaxed);

    let allocs = ALLOCS.load(Ordering::Relaxed);
    assert_eq!(popped, pushed, "the handoff must not lose samples");
    assert_eq!(allocs, 0, "RT handoff performed {allocs} heap allocations");
}
