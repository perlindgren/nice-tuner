// The lighting vocabulary shared by every shader that lights a surface:
// the two field record types, the shadow test that says whether an occluder
// stands between a light and a pixel, and the loop over the lights that
// paints a surface with them.
//
// This file is not a shader. `src/shaders.rs` prepends it to `shape.wgsl`,
// `sprite.wgsl`, `particles.wgsl` and `tilemap.wgsl`, the four
// pipelines that read the frame's light and occluder fields; the other
// shaders get no part of it.
// It declares no `@group` and no `@binding`, because the two storage
// bindings sit at different indices in each of those four: each host
// declares its own `field` and `occluders` variables around this text. A
// module-scope name in WGSL is in scope across the whole program, so a
// function here may read a variable its host declares after it - that
// forward reference is the contract, not an accident, and the naga test in
// `src/shaders.rs` is what keeps it honest.

// The frame's light field: one global storage buffer shared by every lit
// pipeline. The 32-byte header holds the light count, 12 padding bytes,
// and the scene's ambient color; the unsized array tail holds one 64-byte
// record per light, in call order — (x, y, radius, intensity), then
// (r, g, b, penumbra), then (dir_x, dir_y, cos_half, feather-or-taps),
// then (layers, 0, 0, 0):
// the light's color with its shadow penumbra radius in pixels, then the
// cone's unit axis, the cosine of its half opening angle, and the width
// of its edge feather in cosine units (0.0 for a hard edge; omni lights
// pass every gate). The fourth vec4's x is the light's light-layer mask —
// the raw bits of its `LightLayers` set shipped through `f32::from_bits`
// and read back with `bitcast<u32>`, never as a number to do arithmetic
// on — and its other components are padding. A light reaches a receiver
// only when its mask and the receiver's share a layer. The CPU sizes the
// bound buffer to fit the frame's lights.
struct LightField {
    count: u32,
    pad: array<u32, 3>,
    ambient: vec4<f32>,
    lights: array<vec4<f32>>,
};

// The frame's occluder field: one global storage buffer shared by every lit
// pipeline. The 16-byte header holds the record count and the frame's light
// count, then 8 padding bytes; the unsized array tail opens with a range
// table — one (start, end, _, _) vec4 per light, in vec4 units from the
// tail's start — and after the table runs one contiguous block of occluder
// records per light: exactly those whose world bounds reach some ray that
// light fires at a pixel it actually reaches — past a ranged light's radius
// `light_mix` asks no ray at all (the reach reject below), and a ray starts
// on the light's penumbra disk, so the block is the disk of rays that
// remains — so a walk over a light's block can never miss a blocker its own
// contributing rays could meet. Each record is the shape it
// always was — (m00, m10, m01, m11), the inverse of the occluder's world
// transform, column-major (pixel space to the occluder's local space), then
// (tx, ty, cx, cy), the inverse transform's translation and the box's local
// center, then (hx, hy, planes, layers), the box's local half-extents, the
// count of half-planes that follow the record — zero for a box, and for a
// convex polygon one (n.x, n.y, c, _) vec4 per edge, its outward unit normal
// and the offset it sits at, both in the occluder's own space — and the
// occluder's light-layer mask in the w that was padding before layers: raw
// bits through `f32::from_bits`, read back with `bitcast<u32>`. A record
// shadows a receiver only when its mask and the receiver's share a layer,
// whoever the light behind it was. Records are therefore
// variable length, `3 + planes` vec4s apiece. The CPU sizes the bound buffer
// to fit the frame's lights and blocks; an occluder that can shadow several
// lights is copied into every block that can use it.
struct OccluderField {
    count: u32,
    lights: u32,
    pad: array<u32, 2>,
    occluders: array<vec4<f32>>,
};

// Whether a point in an occluder's own space lies inside all `planes` of its
// edges, each held as (n.x, n.y, c, _) with the inside being dot(n, q) <= c.
fn in_planes(base: u32, planes: u32, q: vec2<f32>) -> bool {
    for (var k = 0u; k < planes; k++) {
        let n = occluders.occluders[base + k];
        if (n.x * q.x + n.y * q.y > n.z) {
            return false;
        }
    }
    return true;
}

// Whether the open segment (ll, lp) keeps a piece inside all `planes` edges:
// the box's slab clip with more slabs, because a convex shape *is* the
// intersection of its edges' half-planes. Each endpoint's distance to an edge
// is linear along the segment, so one crossing bounds the interval, and an edge
// with both endpoints outside it ends the search outright — which is why this
// clip needs no epsilon on a direction, where the slab test above needs one.
fn clip_planes(base: u32, planes: u32, ll: vec2<f32>, lp: vec2<f32>) -> bool {
    var tmin = 0.0;
    var tmax = 1.0;
    for (var k = 0u; k < planes; k++) {
        let n = occluders.occluders[base + k];
        let da = n.x * ll.x + n.y * ll.y - n.z;
        let db = n.x * lp.x + n.y * lp.y - n.z;
        if (da > 0.0) {
            if (db > 0.0) {
                return false;
            }
            tmin = max(tmin, da / (da - db));
        } else if (db > 0.0) {
            tmax = min(tmax, da / (da - db));
        }
    }
    return tmax > tmin && tmax > 0.0 && tmin < 1.0;
}

// Whether any occluder blocks the light at `l` from reaching the pixel `p`:
// true when the open segment from the light to the pixel passes through the
// interior of one of them. The test runs in each occluder's own space, where a
// box is axis-aligned about its center — the classic slab interval of a line
// against an axis-aligned box — and a convex polygon is the list of edges it
// was shipped with, clipped against that same interval.
//
// `recv` is the *receiver's* light-layer mask: a record whose own mask shares
// no layer with it casts no shadow here, whatever light is being tested —
// the filter asks "whose surfaces do I shadow", never "which lights do I
// block", so the same wall can shade the far background while the floor it
// stands on sees no shadow from it at all.
//
// Records vary in length, so the walk carries an offset rather than multiplying
// an index: it walks the light's own block in order, since the first blocker
// ends the search, and in exchange a box keeps the exact 48 bytes it had before
// polygons existed. A box carries no edges and the slabs are the whole shape; a
// polygon's half-extents are the box *around* it, so the slabs become the cheap
// reject in front of its edges, which is worth having at up to sixteen clips a
// ray, nine rays a pixel, per light.
fn occluded(li: u32, recv: u32, l: vec2<f32>, p: vec2<f32>) -> bool {
    // Walk only the light's own block: the range table's `start..end` for
    // light `li`, in vec4 units from the tail's start. The packer put into
    // that block every occluder whose bounds reach this light's rays at a
    // pixel the light reaches, and only those — so an empty block answers
    // "nothing stands in the way" without a single record load, and a full
    // one is the walk the global field used to demand of every light,
    // narrowed to the records that could honestly block this ray.
    let range = occluders.occluders[li];
    var base = u32(range.x);
    let end = u32(range.y);
    while (base < end) {
        let rec = occluders.occluders[base + 2u];
        let edges = u32(rec.z);
        // The record's layer mask, in the w that used to be zero padding:
        // one shared layer and the shadow can land on this receiver;
        // none, and the walk steps over the record — its planes included,
        // which is why the offset advances before continuing.
        if (bitcast<u32>(rec.w) & recv) == 0u {
            base += 3u + edges;
            continue;
        }
        let m = occluders.occluders[base];
        let tv = occluders.occluders[base + 1u];
        let half = rec.xy;
        let center = tv.zw;
        // The light and the pixel in the occluder's local space.
        let ll = vec2<f32>(m.x * l.x + m.z * l.y + tv.x, m.y * l.x + m.w * l.y + tv.y);
        let lp = vec2<f32>(m.x * p.x + m.z * p.y + tv.x, m.y * p.x + m.w * p.y + tv.y);
        // The occluder never blocks the light from its own surface: a pixel on
        // or inside it is lit directly, and it shadows only what lies behind.
        // For a polygon the record's extents are only its bounds, so stepping
        // inside the bounds is not yet stepping inside the shape.
        var inside = all(abs(lp - center) <= half);
        if (inside && edges > 0u) {
            inside = in_planes(base + 3u, edges, lp);
        }
        var blocked = false;
        if (!inside) {
            // The interval of t in [0, 1] where the segment ll + t * (lp - ll)
            // lies inside the bounds, built slab by slab.
            let d = lp - ll;
            var tmin = 0.0;
            var tmax = 1.0;
            if (abs(d.x) > 1e-6) {
                let t1 = (center.x - half.x - ll.x) / d.x;
                let t2 = (center.x + half.x - ll.x) / d.x;
                tmin = max(tmin, min(t1, t2));
                tmax = min(tmax, max(t1, t2));
            } else if (ll.x < center.x - half.x || ll.x > center.x + half.x) {
                tmax = -1.0; // parallel to the slab, outside it: no crossing
            }
            if (abs(d.y) > 1e-6) {
                let t1 = (center.y - half.y - ll.y) / d.y;
                let t2 = (center.y + half.y - ll.y) / d.y;
                tmin = max(tmin, min(t1, t2));
                tmax = min(tmax, max(t1, t2));
            } else if (ll.y < center.y - half.y || ll.y > center.y + half.y) {
                tmax = -1.0;
            }
            // The segment cuts through the bounds when the interval has
            // positive length and reaches into the open segment (0, 1): the
            // pixel is strictly outside the occluder, so the exit lands before
            // t = 1. A box stops here; a polygon asks its edges, which are the
            // shape and get the last word.
            if (tmax > tmin && tmax > 0.0 && tmin < 1.0) {
                blocked = edges == 0u || clip_planes(base + 3u, edges, ll, lp);
            }
        }
        if (blocked) {
            return true;
        }
        base += 3u + edges;
    }
    return false;
}

// The shadow-ray taps on the light's penumbra disk: a 3x3 grid in unit
// space, scaled per pixel by the light's penumbra radius (the record's
// `lc.w`). A light of physical size is a disk, not a mathematical point,
// so every tap is its own shadow ray from a point of that disk.
// The same disk sampled four times: the cross of axis points, the
// balanced quarter of the grid for hardware that cannot pay nine.
const SHADOW_TAPS4 = array<vec2<f32>, 4>(
    vec2<f32>(-1.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, -1.0),
    vec2<f32>(0.0, 1.0),
);

const SHADOW_TAPS = array<vec2<f32>, 9>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(0.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
);

// How far back along a directional light's travel axis its virtual
// source sits (in pixels): shadow rays run from there to the pixel, so
// they are near-parallel across a window and the light reads as distant.
// A larger range makes shadows more parallel and the penumbra harder, as
// a source nearer the horizon's infinity would be.

// Which of a penumbra's taps the walk finds blocked, as a bitmask: bit `t` set
// means ray `t` of the disk found a blocker. This is `occluded` walked once
// for the whole disk instead of once per ray, and the reason it is worth the
// extra shape in the file is measured: nine rays with nothing to walk cost
// 0.43 ms across the suite's ten lamps, the same nine rays over sixty
// occluders cost 5.62 ms (`taps-9-clear` against `taps-9` in the offscreen
// meter). The walk is the cliff, and every step of it was being paid nine
// times over for work that does not vary between taps — the record's three
// loads, the pixel's trip through the occluder's transform, and whether the
// pixel sits inside the shape at all. Here each of those happens once per
// record; only the slab interval, which genuinely depends on where on the
// disk the ray started, runs per tap.
//
// It also visits fewer records than the nine separate walks did. Each of
// those stopped at the first blocker for its own ray; this one stops when
// every unresolved ray has found one, which is the longest of the nine
// prefixes rather than their sum. A pixel in deep shadow used to walk the
// same short prefix nine times and now walks it once; a pixel in open light
// walked the whole block nine times and now walks it once with a ninth of the
// loads per step.
//
// `src` is the disk's centre, `rad` its radius in pixels, `ca`/`sa` the
// per-pixel rotation that keeps the sample grid from banding, `taps` how many
// rays sweep it, and `four` selects the balanced cross instead of the grid.
// One ray of the penumbra disk against an occluder's bounds, in the occluder's
// own space: where the ray starts (`ll`) and the interval its slab test found
// along the segment to the pixel. Returned as one vec4 so a caller can hold a
// pair of them in scalars — the intervals are four floats each and an array
// under a loop index is the thing that sends this to private memory.
fn tap_interval(
    t: u32,
    four: bool,
    rad: f32,
    ca: f32,
    sa: f32,
    lc: vec2<f32>,
    lp: vec2<f32>,
    m: vec4<f32>,
    center: vec2<f32>,
    half: vec2<f32>,
) -> vec4<f32> {
    // Branch, not select: both select arms evaluate, and the four-cross must
    // never be indexed past its four.
    var o = SHADOW_TAPS[t];
    if four {
        o = SHADOW_TAPS4[t];
    }
    o = o * rad;
    let q = vec2<f32>(ca * o.x - sa * o.y, sa * o.x + ca * o.y);
    let ll = vec2<f32>(lc.x + m.x * q.x + m.z * q.y, lc.y + m.y * q.x + m.w * q.y);
    let d = lp - ll;
    var tmin = 0.0;
    var tmax = 1.0;
    if (abs(d.x) > 1e-6) {
        let t1 = (center.x - half.x - ll.x) / d.x;
        let t2 = (center.x + half.x - ll.x) / d.x;
        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    } else if (ll.x < center.x - half.x || ll.x > center.x + half.x) {
        tmax = -1.0;
    }
    if (abs(d.y) > 1e-6) {
        let t1 = (center.y - half.y - ll.y) / d.y;
        let t2 = (center.y + half.y - ll.y) / d.y;
        tmin = max(tmin, min(t1, t2));
        tmax = min(tmax, max(t1, t2));
    } else if (ll.y < center.y - half.y || ll.y > center.y + half.y) {
        tmax = -1.0;
    }
    return vec4<f32>(ll.x, ll.y, tmin, tmax);
}

fn occluded_taps(
    li: u32,
    recv: u32,
    src: vec2<f32>,
    p: vec2<f32>,
    rad: f32,
    ca: f32,
    sa: f32,
    taps: u32,
    four: bool,
) -> u32 {
    let every = (1u << taps) - 1u;
    var blocked = 0u;
    // How far a tap's origin can sit from the disk's centre, in pixels: the
    // 3x3 grid reaches sqrt(2) out at its corners, the balanced cross only 1.
    let reach = rad * select(1.0, 1.4142136, !four);
    let range = occluders.occluders[li];
    var base = u32(range.x);
    let end = u32(range.y);
    while (base < end) {
        let rec = occluders.occluders[base + 2u];
        let edges = u32(rec.z);
        if (bitcast<u32>(rec.w) & recv) == 0u {
            base += 3u + edges;
            continue;
        }
        let m = occluders.occluders[base];
        let tv = occluders.occluders[base + 1u];
        let half = rec.xy;
        let center = tv.zw;
        // Once per record, not once per ray: where the pixel and the middle
        // of the disk sit in this occluder's space. Every tap's origin is
        // `lc` plus a rotated offset, so the per-tap transform below is only
        // the offset — the disk is small and the matrix is applied to it, not
        // to nine absolute points.
        let lp = vec2<f32>(m.x * p.x + m.z * p.y + tv.x, m.y * p.x + m.w * p.y + tv.y);
        let lc = vec2<f32>(m.x * src.x + m.z * src.y + tv.x, m.y * src.x + m.w * src.y + tv.y);
        // The occluder never shadows its own surface, and which surface the
        // pixel sits on does not depend on which tap is being asked about.
        var inside = all(abs(lp - center) <= half);
        if (inside && edges > 0u) {
            inside = in_planes(base + 3u, edges, lp);
        }
        // The whole-disk reject, boxes only. Every point of a tap's segment
        // sits within the disk's reach of the matching point of the centre
        // segment — the two differ by (1 - t) * (q - centre), and |q - centre|
        // is at most the reach — so when the centre segment misses this box
        // inflated by that reach, no ray of the disk can touch it and the nine
        // slab tests below would answer "not here" nine times over. In the
        // occluder's own space the disk is an ellipse, so the inflation is the
        // reach scaled by the transform's longest column, which bounds the
        // local offset from above whatever the disk's orientation turns out to
        // be. It reads the record's bounds, and that is sound for a polygon
        // too: the packer ships a polygon's own bounding box as those bounds,
        // so a disk that misses the box misses the shape, whatever the edges
        // then have to say about a disk that reaches it.
        if (reach > 0.0) {
            let hx = half.x + reach * sqrt(m.x * m.x + m.z * m.z);
            let hy = half.y + reach * sqrt(m.y * m.y + m.w * m.w);
            let dc = lp - lc;
            var rmin = 0.0;
            var rmax = 1.0;
            if (abs(dc.x) > 1e-6) {
                let a1 = (center.x - hx - lc.x) / dc.x;
                let a2 = (center.x + hx - lc.x) / dc.x;
                rmin = max(rmin, min(a1, a2));
                rmax = min(rmax, max(a1, a2));
            } else if (lc.x < center.x - hx || lc.x > center.x + hx) {
                rmax = -1.0;
            }
            if (abs(dc.y) > 1e-6) {
                let a1 = (center.y - hy - lc.y) / dc.y;
                let a2 = (center.y + hy - lc.y) / dc.y;
                rmin = max(rmin, min(a1, a2));
                rmax = min(rmax, max(a1, a2));
            } else if (lc.y < center.y - hy || lc.y > center.y + hy) {
                rmax = -1.0;
            }
            if (!(rmax > rmin && rmax > 0.0 && rmin < 1.0)) {
                base += 3u + edges;
                continue;
            }
        }
        if (!inside) {
            // Two rays at a time, in scalars. The plane list is the expensive
            // part of a shape and it is walked once per pair instead of once
            // per ray — five walks for nine rays, and the pixel's side of each
            // edge is computed once per walk. Nine intervals carried in an
            // array cost more than the walks they saved (§22), so the state
            // here is four floats at a time and nothing else.
            var t = 0u;
            while (t < taps) {
                let a = tap_interval(t, four, rad, ca, sa, lc, lp, m, center, half);
                let u = t + 1u;
                var b = vec4<f32>(0.0);
                if (u < taps) {
                    b = tap_interval(u, four, rad, ca, sa, lc, lp, m, center, half);
                }
                let a_open = ((blocked >> t) & 1u) == 0u && a.w > a.z && a.w > 0.0 && a.z < 1.0;
                let b_open = u < taps && ((blocked >> u) & 1u) == 0u && b.w > b.z && b.w > 0.0
                    && b.z < 1.0;
                if (edges == 0u) {
                    // A box's bounds are the whole shape: the slab above is
                    // the entire answer, exactly as it was before shapes.
                    if (a_open) {
                        blocked |= 1u << t;
                    }
                    if (b_open) {
                        blocked |= 1u << u;
                    }
                } else if (a_open || b_open) {
                    var ka = a_open;
                    var kb = b_open;
                    var amin = a.z;
                    var amax = a.w;
                    var bmin = b.z;
                    var bmax = b.w;
                    for (var k = 0u; k < edges; k++) {
                        let n = occluders.occluders[base + 3u + k];
                        let db = n.x * lp.x + n.y * lp.y - n.z;
                        if (ka) {
                            let da = n.x * a.x + n.y * a.y - n.z;
                            if (da > 0.0) {
                                if (db > 0.0) {
                                    ka = false;
                                } else {
                                    amin = max(amin, da / (da - db));
                                }
                            } else if (db > 0.0) {
                                amax = min(amax, da / (da - db));
                            }
                        }
                        if (kb) {
                            let da = n.x * b.x + n.y * b.y - n.z;
                            if (da > 0.0) {
                                if (db > 0.0) {
                                    kb = false;
                                } else {
                                    bmin = max(bmin, da / (da - db));
                                }
                            } else if (db > 0.0) {
                                bmax = min(bmax, da / (da - db));
                            }
                        }
                        if (!ka && !kb) {
                            break;
                        }
                    }
                    if (ka && amax > amin && amax > 0.0 && amin < 1.0) {
                        blocked |= 1u << t;
                    }
                    if (kb && bmax > bmin && bmax > 0.0 && bmin < 1.0) {
                        blocked |= 1u << u;
                    }
                }
                t = t + 2u;
            }
        }
        base += 3u + edges;
        if (blocked == every) {
            break;
        }
    }
    return blocked;
}

// The light mix at a pixel: the scene's ambient plus every light's
// contribution — the light's color times its intensity, scaled by the
// quadratic falloff from the light's position out to its radius, zero
// outside the light's cone — feathered to zero across its soft edges —
// and ramped to zero across its penumbra where an occluder blocks the
// light's path to the pixel. The result multiplies the unlit color:
// pixel = base * (ambient + lights).
//
// `recv` is the receiver's own light-layer mask, and it filters both ends
// of the mix: a light whose mask shares no layer with it never contributes,
// and an occluder whose mask shares no layer with it never shadows it.
// Ambient is no one's light: it always applies, whatever the mask.

const SHADOW_RANGE: f32 = 1500.0;
fn light_mix(p: vec2<f32>, recv: u32) -> vec3<f32> {
    var c = field.ambient.rgb;
    for (var i = 0u; i < field.count; i++) {
        // The light's layer mask, in the fourth vec4's x: raw bits, folded
        // against the receiver's. No shared layer means the light simply
        // is not shining here — skipped before a single distance is taken,
        // so an off-layer light costs one load and a compare.
        if (bitcast<u32>(field.lights[4 * i + 3u].x) & recv) == 0u {
            continue;
        }
        let l = field.lights[4 * i];
        let lc = field.lights[4 * i + 1];
        let cone = field.lights[4 * i + 2];
        // A directional light is packed with a negative radius (see
        // `Light::directional`): it has no position and no falloff, so
        // every shadow ray runs parallel to the travel axis `cone.xy`
        // from a virtual source `src` placed `SHADOW_RANGE` back along
        // that axis through `p`. Everything downstream — the gate, the
        // penumbra taps, the occlusion ray — then treats `src` as if it
        // were an ordinary point light's position, so the parallel case
        // costs nothing new.
        let directional = l.z < 0.0;
        // The reach reject, and the falloff it rides on. A point or cone
        // light (radius >= 0) contributes `f * f` of itself to a pixel,
        // and past its radius `f` is zero — so the loop used to walk the
        // light's whole occluder block for pixels the light could not
        // touch even a fraction of, paying every slab test to multiply
        // the answer by nothing. A squared-distance test stands in front
        // of the entire body instead: a pixel outside a lamp's reach
        // asks that lamp nothing at all. `rr` carries the `max` the
        // falloff always applied — a light of radius under 1e-4 still
        // lights the pixel it sits on, and the reject agrees with the
        // old `f` to the last ulp of the radius — it skips only where
        // the old contribution was zero. Skipped with it is the light's
        // whole arithmetic, gate and shadow rays included — which is the
        // whole point: the lamps of a lit scene reach a fraction of the
        // screen each, and every pixel outside every one of them used to
        // walk every one of their occluder blocks to add nothing.
        var f = 1.0;
        if (!directional) {
            let to = p - l.xy;
            let d2 = dot(to, to);
            let rr = max(l.z, 1e-4);
            if (d2 >= rr * rr) {
                continue;
            }
            f = 1.0 - sqrt(d2) / rr;
        }
        let src = select(l.xy, p - cone.xy * SHADOW_RANGE, directional);
        // The cone gate, as a 0..1 factor: omni and unfeathered cones
        // keep the hard test, without normalizing d so the pixel at the
        // light's own position stays inside the cone. An omni light's
        // cos_half is -1.0, which every pixel passes.
        let d = p - src;
        let len = length(d);
        let ad = dot(d, cone.xy);
        var gate = select(0.0, 1.0, ad >= cone.z * len);
        // A feathered cone (feather > 0) instead ramps the gate smoothly
        // from 0 at the cone edge up to 1 once the direction is `cone.w`
        // of cosine inside it — a soft edge in place of the hard cut.
        // The ramp needs d normalized; the light's own pixel keeps the
        // hard gate's answer above.
        if (cone.w > 0.0 && cone.z > -1.0 && len > 1e-4) {
            gate = smoothstep(cone.z, cone.z + cone.w, ad / len);
        }
        if (gate > 0.0) {
            // How much of the light reaches this pixel past the
            // occluders. Outside the cone nothing contributes anyway, so
            // the raycasts are skipped entirely there. A penumbra radius
            // of 0 (a point-sized light) keeps the single hard ray;
            // otherwise nine rays issue from the light's disk and the
            // unoccluded fraction lights the pixel — the shadow's edge
            // ramps across the disk's width and widens the farther it
            // falls behind the occluder, like a real light patch. The
            // tap grid rotates per pixel (interleaved-gradient angle) so
            // the ten sample levels scatter into fine dither instead of
            // stacking into bands.
            var v = 1.0;
            if (lc.w > 0.0) {
                let ang = 6.2831853
                    * fract(52.9829 * fract(0.06711 * p.x + 0.005837 * p.y));
                let sa = sin(ang);
                let ca = cos(ang);
                // How many rays sweep the disk. Omni (and directional)
                // lights carry the rung in the slot a cone uses for
                // its feather: 0 is the full nine — every light that
                // never named a rung — and 1..=9 walks that many of
                // the grid, four of them from the balanced cross.
                var taps = 9u;
                if (cone.z <= -1.0 && cone.w > 0.0) {
                    taps = clamp(u32(cone.w), 1u, 9u);
                }
                // One walk for the whole disk: the mask says which taps
                // found a blocker, and the lit fraction is the rest — the
                // same ten sample levels the nine separate rays produced,
                // gathered without walking the block nine times.
                let blocked = occluded_taps(i, recv, src, p, lc.w, ca, sa, taps, taps <= 4u);
                v = f32(taps - countOneBits(blocked)) / f32(taps);
            } else {
                // A point-sized light: the single ray from its position,
                // the original hard shadow. Without this branch the
                // default above would stand and the light would ignore
                // every occluder.
                v = select(0.0, 1.0, !occluded(i, recv, src, p));
            }
            c += lc.rgb * l.w * f * f * v * gate;
        }
    }
    return c;
}
