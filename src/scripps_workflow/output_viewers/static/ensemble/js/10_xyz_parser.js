// ---- Multi-frame XYZ parsing ----------------------------------------

function parse_multi_frame_xyz(text) {
    // Returns [{index, n_atoms, comment, text, energy_hartree}].
    // ``text`` per frame is a fully self-contained xyz block
    // (n_atoms\ncomment\natom_lines\n) suitable for 3Dmol.addModel.
    var frames = [];
    var lines = text.split(/\r?\n/);
    var i = 0;
    while (i < lines.length) {
        // Skip leading blank lines between frames.
        while (i < lines.length && lines[i].trim() === '') i++;
        if (i >= lines.length) break;

        var n_atoms = parseInt(lines[i].trim(), 10);
        if (!Number.isFinite(n_atoms) || n_atoms <= 0) {
            console.warn('parse_multi_frame_xyz: bad atom count at line', i, ':', lines[i]);
            i++;
            continue;
        }
        var comment = (i + 1 < lines.length ? lines[i + 1] : '').trim();
        var frame_lines = lines.slice(i, i + 2 + n_atoms);
        if (frame_lines.length < 2 + n_atoms) {
            console.warn('parse_multi_frame_xyz: truncated frame at index', frames.length);
            break;
        }
        var frame_text = frame_lines.join('\n') + '\n';

        // ORCA convention: "Coordinates from orca-job orca_opt E -114.299..."
        // CREST writes the energy alone on the line. Match either; fall
        // back to null if neither.
        var energy_hartree = null;
        var m = comment.match(/\bE\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)/);
        if (m) {
            energy_hartree = parseFloat(m[1]);
        } else {
            // CREST-style: just the energy on its own (no "E " prefix).
            var bare = comment.match(/^\s*(-?\d+\.\d+(?:[eE][+-]?\d+)?)\s*$/);
            if (bare) energy_hartree = parseFloat(bare[1]);
        }

        frames.push({
            index: frames.length + 1,
            n_atoms: n_atoms,
            comment: comment,
            text: frame_text,
            energy_hartree: energy_hartree,
        });
        i += 2 + n_atoms;
    }
    return frames;
}
