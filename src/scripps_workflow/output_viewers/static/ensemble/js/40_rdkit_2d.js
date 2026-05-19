// ---- SMILES block + 2D structure ----------------------------------

function render_smiles_block() {
    var state = window.__viewer_state;
    if (!state) return;
    var block = document.getElementById('smiles-block');
    var val = document.getElementById('smiles-value');
    if (!block || !val) return;
    if (state.smiles) {
        val.textContent = state.smiles;
        block.hidden = false;
    } else {
        val.textContent = '';
        block.hidden = true;
    }
}

function render_2d_structure() {
    var state = window.__viewer_state;
    if (!state || !state.smiles) return;
    var box = document.getElementById('structure-2d');
    var host = document.getElementById('structure-canvas');
    if (!box || !host) return;
    if (!window.__rdkit) {
        // RDKit's WASM init hasn't finished yet. kick_off_rdkit()'s
        // .then will re-call render_2d_structure once it's ready.
        return;
    }
    var mol = null;
    try {
        // PREFERRED: parse the first xyz frame as a mol so the
        // resulting atom indices match the xyz file's atom order
        // exactly (no SMILES→XYZ remapping needed). Some RDKit JS
        // builds don't accept xyz blocks via get_mol — in that case
        // we fall back to the SMILES parse + the ad-hoc atom map
        // (TEST_FIXTURE_ATOM_MAPS). state.uses_xyz_mol records
        // which path we ended up on so align_to_selection knows
        // whether to apply the map.
        if (Array.isArray(state.atom_map)) {
            // Preferred standalone-bundle path: the Python bundle builder
            // already derived a SMILES-index -> XYZ-index atom map using
            // full RDKit. Draw from SMILES for clean 2D coordinates and use
            // the supplied map for 2D selection -> 3D alignment. This avoids
            // relying on RDKit.js to perceive bonds directly from raw XYZ,
            // which is not consistently supported in the browser build.
            state.uses_xyz_mol = false;
            mol = window.__rdkit.get_mol(state.smiles);
        } else {
            mol = try_build_mol_from_xyz_first_frame(state);
            if (mol) {
                state.uses_xyz_mol = true;
            } else {
                state.uses_xyz_mol = false;
                mol = window.__rdkit.get_mol(state.smiles);
            }
        }
        if (!mol || !mol.is_valid()) {
            console.warn('RDKit: invalid SMILES:', state.smiles);
            return;
        }
        // Build the highlights JSON from the current selection.
        // get_svg_with_highlights accepts:
        //   atoms / bonds: indices to highlight
        //   highlightAtomColors / highlightBondColors: maps to RGB
        // Use a single accent color (light blue) so it's visually
        // distinct from RDKit's default heteroatom colors.
        var highlight_atoms = Object.keys(state.selected_atoms || {})
            .map(function(k) { return parseInt(k, 10); });
        var highlight_bonds = derive_highlight_bonds(mol, state.selected_atoms);

        var color = [0.18, 0.52, 0.92];  // #2e85eb-ish (matches the active conformer accent)
        var atomColors = {};
        var bondColors = {};
        highlight_atoms.forEach(function(a) { atomColors[a] = color; });
        highlight_bonds.forEach(function(b) { bondColors[b] = color; });

        var opts = JSON.stringify({
            atoms: highlight_atoms,
            bonds: highlight_bonds,
            highlightAtomColors: atomColors,
            highlightBondColors: bondColors,
            highlightAtomRadii: {},  // default
        });
        var svg = mol.get_svg_with_highlights(opts);
        host.innerHTML = svg;
        // Critical ordering: unhide the inset BEFORE expanding hit
        // areas. expand_svg_hitboxes calls getBBox() on labeled-atom
        // SVG elements to find their actual center, but getBBox()
        // returns zeroes for elements whose ancestor is display:none.
        // If we ran the hitbox pass while box was still hidden, the
        // bond-endpoint fallback would fire and the O hitbox would
        // land at the label edge instead of the label center — visible
        // as "the O is only clickable on its second selection" since
        // subsequent re-renders happen with box visible.
        box.hidden = false;
        // Force a synchronous layout pass so getBBox returns real
        // values rather than zeros for newly-attached elements.
        void box.offsetHeight;
        // Add invisible thicker hit areas for bonds + click circles
        // for unlabeled atoms so the user doesn't have to land on
        // RDKit's thin paths perfectly.
        var svg_root = host.querySelector('svg');
        if (svg_root) expand_svg_hitboxes(svg_root);
        // Re-attach delegated click handler on host (innerHTML wipes
        // any previous listeners on inner SVG nodes; delegation here
        // means we only need one listener regardless of re-renders).
        install_structure_click_handler(host);
        update_selection_toolbar();
    } catch (err) {
        console.warn('RDKit render failed:', err);
    } finally {
        if (mol) {
            try { mol.delete(); } catch (e) { /* ignore */ }
        }
    }
}

function derive_highlight_bonds(mol, selected_atoms) {
    // A bond is highlighted iff both its endpoints are in the
    // selected_atoms set. We get the bond list from RDKit's
    // get_molblock(); each bond line has the two atom indices.
    if (!selected_atoms || Object.keys(selected_atoms).length === 0) {
        return [];
    }
    var bonds = [];
    try {
        var mb = mol.get_molblock();
        // V2000 MOL: counts line has "<natoms> <nbonds> ..." then
        // natoms atom lines, then nbonds bond lines of the form
        // "<a1> <a2> <bond_type> ...". 1-indexed atoms.
        var lines = mb.split('\n');
        var counts = lines[3].trim().split(/\s+/);
        var n_atoms = parseInt(counts[0], 10);
        var n_bonds = parseInt(counts[1], 10);
        for (var i = 0; i < n_bonds; i++) {
            var line = lines[4 + n_atoms + i];
            // Bond line columns are width-3 each (V2000); fall back
            // to whitespace split if the format is non-standard.
            var a1, a2;
            if (line.length >= 6 && /^\s*\d/.test(line)) {
                a1 = parseInt(line.substring(0, 3), 10) - 1;
                a2 = parseInt(line.substring(3, 6), 10) - 1;
            } else {
                var parts = line.trim().split(/\s+/);
                a1 = parseInt(parts[0], 10) - 1;
                a2 = parseInt(parts[1], 10) - 1;
            }
            if (selected_atoms[a1] && selected_atoms[a2]) bonds.push(i);
        }
    } catch (e) {
        console.warn('derive_highlight_bonds failed:', e);
    }
    return bonds;
}

function install_structure_click_handler(host) {
    if (host.__click_installed) return;
    host.__click_installed = true;
    host.addEventListener('click', function(e) {
        var t = e.target;
        if (!t || !t.getAttribute) return;
        // RDKit's class string can carry several tokens: a bond path
        // has "bond-N atom-A atom-B". Parse all atom-* tokens; if
        // there are two, treat as bond click → toggle both atoms.
        // If exactly one, it's a heteroatom label → toggle that atom.
        var cls = t.getAttribute('class') || '';
        var atom_idx = [];
        cls.split(/\s+/).forEach(function(token) {
            var m = token.match(/^atom-(\d+)$/);
            if (m) atom_idx.push(parseInt(m[1], 10));
        });
        if (atom_idx.length === 0) return;
        e.stopPropagation();
        toggle_atoms(atom_idx);
    });
}

function structure_bg_dblclick(ev) {
    // Double-clicking the inset background EXPANDS to the center
    // overlay (a quick alternative to clicking the chevron). It
    // does NOT collapse — collapse is chevron-only, so a stray
    // double-click on the canvas while picking atoms can't accidentally
    // tear down the expanded view. Atom/bond hits are ignored (they
    // pass through to selection logic).
    var t = ev.target;
    if (t && t.getAttribute) {
        var cls = t.getAttribute('class') || '';
        if (/(?:^|\s)(atom-|bond-)/.test(cls)) return;
    }
    var state = window.__viewer_state;
    if (!state || state.is_expanded) return;
    toggle_structure_expanded(ev);
}

function expand_svg_hitboxes(svg_root) {
    // RDKit's bonds are thin strokes and unlabeled atoms (carbons)
    // have no clickable surface at all. Add invisible "hit area"
    // overlays so the user can click anywhere near a bond/atom.
    //
    // Bonds: clone each bond path with a transparent thicker stroke,
    // insert BEFORE the visible bond in the DOM (so visible art
    // paints over the hitbox), and tag pointer-events on the hit
    // path. Carries the same class string so the click delegation
    // handler treats it identically.
    //
    // Atoms: extract each atom's 2D position from the first bond
    // path endpoint that references it, then add an invisible
    // <circle> of radius 8 (SVG units) tagged with class="atom-N".
    if (!svg_root || svg_root.__hitboxes_added) return;
    svg_root.__hitboxes_added = true;

    var ns = 'http://www.w3.org/2000/svg';
    var bonds = svg_root.querySelectorAll('[class*="bond-"]');

    var atom_pos = {};  // atom_idx → [x, y]
    bonds.forEach(function(b) {
        var cls = b.getAttribute('class') || '';
        var ais = [];
        cls.split(/\s+/).forEach(function(tok) {
            var m = tok.match(/^atom-(\d+)$/);
            if (m) ais.push(parseInt(m[1], 10));
        });
        var d = b.getAttribute('d') || '';
        // Pull numeric tokens from the d attr; bond paths typically
        // start with "M x1,y1 L x2,y2" but double bonds add a second
        // segment. First two pairs are the bond endpoints.
        var nums = d.match(/-?\d+(?:\.\d+)?/g) || [];
        if (ais.length === 2 && nums.length >= 4) {
            if (atom_pos[ais[0]] === undefined) {
                atom_pos[ais[0]] = [parseFloat(nums[0]), parseFloat(nums[1])];
            }
            if (atom_pos[ais[1]] === undefined) {
                atom_pos[ais[1]] = [parseFloat(nums[2]), parseFloat(nums[3])];
            }
        }
        // Insert invisible thick-stroke clone behind the visible bond
        var hit = b.cloneNode(false);
        hit.setAttribute('stroke', 'transparent');
        hit.setAttribute('stroke-width', '14');
        hit.setAttribute('fill', 'none');
        hit.setAttribute('pointer-events', 'stroke');
        hit.removeAttribute('style');
        b.parentNode.insertBefore(hit, b);
    });

    // For atoms RDKit renders with a visible label (heteroatoms like
    // O, N, S, …), bond paths are TRUNCATED to the label's bounding
    // box, so the bond-endpoint positions above are offset from the
    // true atom centers. Override using the actual rendered bbox of
    // each labeled atom element — this works regardless of how
    // RDKit nests text (sometimes <text class="atom-N">, sometimes
    // <g class="atom-N"><text>...</text></g>, sometimes split across
    // <tspan>s for subscripts).
    var atom_classed = svg_root.querySelectorAll('[class*="atom-"]');
    atom_classed.forEach(function(el) {
        var cls = el.getAttribute('class') || '';
        if (/(?:^|\s)(atom-hitbox|bond-)/.test(cls)) return;
        var atom_tokens = cls.match(/atom-\d+/g) || [];
        // Bonds have two atom-* tokens (atom-A atom-B); atoms have one.
        if (atom_tokens.length !== 1) return;
        var idx = parseInt(atom_tokens[0].replace('atom-', ''), 10);
        var bbox;
        try {
            bbox = el.getBBox();
        } catch (e) {
            return;
        }
        if (bbox && bbox.width > 0 && bbox.height > 0) {
            atom_pos[idx] = [bbox.x + bbox.width / 2,
                             bbox.y + bbox.height / 2];
        }
    });

    // Add invisible click circles per atom. Radius 11 (up from 9) so
    // labeled-atom glyphs are fully covered by the hit area.
    Object.keys(atom_pos).forEach(function(idx) {
        var p = atom_pos[idx];
        var c = document.createElementNS(ns, 'circle');
        c.setAttribute('cx', p[0]);
        c.setAttribute('cy', p[1]);
        c.setAttribute('r', '11');
        c.setAttribute('fill', 'transparent');
        c.setAttribute('pointer-events', 'all');
        c.setAttribute('class', 'atom-' + idx + ' atom-hitbox');
        svg_root.appendChild(c);
    });
}

function toggle_atoms(indices) {
    var state = window.__viewer_state;
    if (!state) return;
    // If every index in `indices` is already selected, toggle them
    // ALL off (so clicking a fully-selected bond clears it).
    // Otherwise, add the ones that aren't.
    var all_set = indices.every(function(i) { return state.selected_atoms[i]; });
    indices.forEach(function(i) {
        if (all_set) {
            delete state.selected_atoms[i];
        } else {
            state.selected_atoms[i] = true;
        }
    });
    state.selection_count = Object.keys(state.selected_atoms).length;
    render_2d_structure();
}

function update_selection_toolbar() {
    var state = window.__viewer_state;
    if (!state) return;
    var count_el = document.getElementById('structure-selection-count');
    var align_btn = document.querySelector('.sb-action.align');
    var reset_btn = document.querySelector('.sb-action.reset');
    var n = state.selection_count || 0;
    if (count_el) {
        count_el.textContent = n === 0
            ? 'No atoms selected'
            : n + ' atom' + (n === 1 ? '' : 's') + ' selected';
    }
    if (align_btn) {
        // Three alignment modes by selection size:
        //   n=1 — pure translation (move the atom to match reference)
        //   n=2 — translation of midpoint + minimal axis rotation of bond
        //   n≥3 — full Kabsch
        align_btn.disabled = n < 1;
        align_btn.title = (n < 1)
            ? 'Select at least one atom to compute alignment'
            : ('Apply ' +
               (n === 1 ? 'translation' :
                n === 2 ? 'translation + axis rotation' :
                'Kabsch') + ' alignment to all conformers');
    }
    if (reset_btn) {
        reset_btn.disabled = (n === 0) && !state.is_aligned;
    }
}
