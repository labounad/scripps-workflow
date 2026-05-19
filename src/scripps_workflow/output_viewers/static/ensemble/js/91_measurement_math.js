// ---- Measurement math utilities ----------------------------------

function vec3(x, y, z) { return { x: x, y: y, z: z }; }
function vsub(a, b) { return vec3(a.x - b.x, a.y - b.y, a.z - b.z); }
function vadd(a, b) { return vec3(a.x + b.x, a.y + b.y, a.z + b.z); }
function vscale(a, s) { return vec3(a.x * s, a.y * s, a.z * s); }
function vdot(a, b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
function vcross(a, b) {
    return vec3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}
function vnorm(a) { return Math.sqrt(vdot(a, a)); }
function vnormalize(a) {
    var n = vnorm(a);
    return n > 1e-12 ? vscale(a, 1 / n) : vec3(0, 0, 0);
}
function vdistance(a, b) { return vnorm(vsub(a, b)); }
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function radians_to_degrees(x) { return x * 180 / Math.PI; }
function format_degrees(x) { return (Math.abs(x) < 0.0005 ? 0 : x).toFixed(2) + '\u00b0'; }
function format_distance(x) { return x.toFixed(3) + ' \u00c5'; }

function angle_abc(a, b, c) {
    var ba = vnormalize(vsub(a, b));
    var bc = vnormalize(vsub(c, b));
    return radians_to_degrees(Math.acos(clamp(vdot(ba, bc), -1, 1)));
}

function dihedral_abcd(a, b, c, d) {
    // Conventional signed torsion angle A-B-C-D, in degrees.
    var b0 = vscale(vsub(b, a), -1);
    var b1 = vsub(c, b);
    var b2 = vsub(d, c);
    var b1n = vnormalize(b1);
    var v = vsub(b0, vscale(b1n, vdot(b0, b1n)));
    var w = vsub(b2, vscale(b1n, vdot(b2, b1n)));
    var x = vdot(v, w);
    var y = vdot(vcross(b1n, v), w);
    return radians_to_degrees(Math.atan2(y, x));
}
