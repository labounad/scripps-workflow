// ---- Align / Reset actions ----------------------------------------

function compute_translation_transform(P_pt, Q_pt) {
    // Single-atom alignment: R = identity, t = P - Q. Applied to all
    // atoms of the target, this shifts the molecule so Q[0] lands on
    // P[0] with orientation unchanged.
    return {
        R: [[1,0,0],[0,1,0],[0,0,1]],
        t: [P_pt[0] - Q_pt[0], P_pt[1] - Q_pt[1], P_pt[2] - Q_pt[2]],
    };
}

function compute_two_atom_transform(P, Q) {
    // Two-atom alignment: translate Q's midpoint to P's midpoint,
    // then apply the *minimal* rotation that aligns the Q-bond
    // vector with the P-bond vector. Endpoints don't get flipped —
    // P[0] ↔ Q[0] and P[1] ↔ Q[1] specifically.
    var p_mid = [(P[0][0]+P[1][0])/2, (P[0][1]+P[1][1])/2, (P[0][2]+P[1][2])/2];
    var q_mid = [(Q[0][0]+Q[1][0])/2, (Q[0][1]+Q[1][1])/2, (Q[0][2]+Q[1][2])/2];
    var p_vec = [P[1][0]-P[0][0], P[1][1]-P[0][1], P[1][2]-P[0][2]];
    var q_vec = [Q[1][0]-Q[0][0], Q[1][1]-Q[0][1], Q[1][2]-Q[0][2]];
    var p_len = Math.sqrt(p_vec[0]*p_vec[0] + p_vec[1]*p_vec[1] + p_vec[2]*p_vec[2]);
    var q_len = Math.sqrt(q_vec[0]*q_vec[0] + q_vec[1]*q_vec[1] + q_vec[2]*q_vec[2]);
    if (p_len < 1e-10 || q_len < 1e-10) {
        // Picks coincide; degrade to translation only.
        return {
            R: [[1,0,0],[0,1,0],[0,0,1]],
            t: [p_mid[0]-q_mid[0], p_mid[1]-q_mid[1], p_mid[2]-q_mid[2]],
        };
    }
    var pu = [p_vec[0]/p_len, p_vec[1]/p_len, p_vec[2]/p_len];
    var qu = [q_vec[0]/q_len, q_vec[1]/q_len, q_vec[2]/q_len];
    // Rotation taking qu → pu via Rodrigues' formula.
    // axis = qu × pu (rotate FROM qu TO pu so the cross is taken in that order),
    // sin(θ) = |axis|, cos(θ) = qu · pu.
    var axis = [
        qu[1]*pu[2] - qu[2]*pu[1],
        qu[2]*pu[0] - qu[0]*pu[2],
        qu[0]*pu[1] - qu[1]*pu[0],
    ];
    var sin_t = Math.sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    var cos_t = qu[0]*pu[0] + qu[1]*pu[1] + qu[2]*pu[2];
    var R;
    if (sin_t < 1e-10) {
        // qu and pu are (anti-)parallel.
        if (cos_t > 0) {
            R = [[1,0,0],[0,1,0],[0,0,1]];
        } else {
            // 180° rotation about any axis perpendicular to qu.
            var seed = Math.abs(qu[0]) < 0.9 ? [1,0,0] : [0,1,0];
            var d = seed[0]*qu[0] + seed[1]*qu[1] + seed[2]*qu[2];
            var perp = [seed[0]-d*qu[0], seed[1]-d*qu[1], seed[2]-d*qu[2]];
            var pn = Math.sqrt(perp[0]*perp[0]+perp[1]*perp[1]+perp[2]*perp[2]);
            var u = [perp[0]/pn, perp[1]/pn, perp[2]/pn];
            R = [
                [2*u[0]*u[0]-1, 2*u[0]*u[1],   2*u[0]*u[2]  ],
                [2*u[1]*u[0],   2*u[1]*u[1]-1, 2*u[1]*u[2]  ],
                [2*u[2]*u[0],   2*u[2]*u[1],   2*u[2]*u[2]-1],
            ];
        }
    } else {
        var k = [axis[0]/sin_t, axis[1]/sin_t, axis[2]/sin_t];
        var C = 1 - cos_t;
        R = [
            [cos_t + k[0]*k[0]*C,    k[0]*k[1]*C - k[2]*sin_t, k[0]*k[2]*C + k[1]*sin_t],
            [k[1]*k[0]*C + k[2]*sin_t, cos_t + k[1]*k[1]*C,    k[1]*k[2]*C - k[0]*sin_t],
            [k[2]*k[0]*C - k[1]*sin_t, k[2]*k[1]*C + k[0]*sin_t, cos_t + k[2]*k[2]*C   ],
        ];
    }
    var Rq = [
        R[0][0]*q_mid[0] + R[0][1]*q_mid[1] + R[0][2]*q_mid[2],
        R[1][0]*q_mid[0] + R[1][1]*q_mid[1] + R[1][2]*q_mid[2],
        R[2][0]*q_mid[0] + R[2][1]*q_mid[1] + R[2][2]*q_mid[2],
    ];
    return {
        R: R,
        t: [p_mid[0]-Rq[0], p_mid[1]-Rq[1], p_mid[2]-Rq[2]],
    };
}

function align_to_selection() {
    var state = window.__viewer_state;
    if (!state) return;
    var n = state.selection_count || 0;
    if (n < 1) return;
    ensure_coords_cache();

    // SMILES-space atom indices (from the 2D RDKit drawing).
    var smiles_indices = Object.keys(state.selected_atoms)
        .map(function(k) { return parseInt(k, 10); })
        .filter(function(i) { return Number.isFinite(i); });

    // Translate the 2D-drawing atom indices to xyz-file atom indices.
    // Three cases:
    //   * uses_xyz_mol === true: drawing was made from the xyz, so
    //     atom-N in the SVG IS xyz atom N. Identity map.
    //   * SMILES-based drawing with a registered ad-hoc map: apply
    //     that map (per the test fixture).
    //   * SMILES-based drawing with no map: assume identity (correct
    //     for workflow-produced ensembles where smiles_to_3d
    //     preserves RDKit's atom order through to xyz).
    var atom_map = state.uses_xyz_mol
        ? null
        : (Array.isArray(state.atom_map) ? state.atom_map : get_atom_map_for_smiles(state.smiles));
    var xyz_indices = smiles_indices.map(function(i) {
        return atom_map ? atom_map[i] : i;
    });
    if (xyz_indices.some(function(i) { return i === undefined || i === null; })) {
        console.warn('Align: atom map missing some indices', smiles_indices);
        return;
    }

    var ref_coords = state.original_coords[state.active_idx];
    var P = xyz_indices.map(function(i) { return ref_coords[i]; });
    if (P.some(function(p) { return !p; })) {
        console.warn('Align: some xyz indices out of range for active conformer');
        return;
    }

    var mode = (n === 1) ? 'translation'
             : (n === 2) ? 'two-atom'
             : 'kabsch';
    console.log('Align (' + mode + ') with smiles idx ' +
                JSON.stringify(smiles_indices) +
                ' → xyz idx ' + JSON.stringify(xyz_indices));

    state.frames.forEach(function(_, frame_idx) {
        var model = state.viewer.getModel(frame_idx);
        // First, restore the model to its original coords so we're
        // computing the transform from a known baseline.
        set_model_coords(model, state.original_coords[frame_idx]);
        if (frame_idx === state.active_idx) return;
        var Q = xyz_indices.map(function(i) {
            return state.original_coords[frame_idx][i];
        });
        if (Q.some(function(q) { return !q; })) return;
        var tr;
        if (mode === 'translation')  tr = compute_translation_transform(P[0], Q[0]);
        else if (mode === 'two-atom') tr = compute_two_atom_transform(P, Q);
        else                          tr = compute_kabsch_transform(P, Q);
        apply_transform_inplace(model, tr.R, tr.t);
    });

    state.is_aligned = true;
    // repaint_viewer() forces 3Dmol to re-derive bond geometry from
    // the new atom positions; calling viewer.render() alone left
    // bonds drawn at old positions until you toggled conformer.
    repaint_viewer();
    update_selection_toolbar();
}

function reset_alignment() {
    var state = window.__viewer_state;
    if (!state) return;
    // Restore original coordinates if we have them.
    if (state.original_coords) {
        state.frames.forEach(function(_, frame_idx) {
            set_model_coords(
                state.viewer.getModel(frame_idx),
                state.original_coords[frame_idx]
            );
        });
        // Force 3Dmol to re-derive bond geometry from the restored
        // positions instead of just redrawing atoms in place.
        repaint_viewer();
    }
    // Clear selection.
    state.selected_atoms = {};
    state.selection_count = 0;
    state.is_aligned = false;
    render_2d_structure();
}
