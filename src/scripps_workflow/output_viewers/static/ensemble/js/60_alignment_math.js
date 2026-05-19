// ---- Kabsch alignment math ----------------------------------------

function ensure_coords_cache() {
    // First call: snapshot every atom's xyz in every model. Used by
    // align_to_selection (as the source) and reset_alignment (as the
    // restore target). Cheap memory-wise — N_frames * N_atoms * 3
    // doubles, typically << 1 MB.
    var state = window.__viewer_state;
    if (!state || state.original_coords) return;
    var cached = [];
    state.frames.forEach(function(_, frame_idx) {
        var model = state.viewer.getModel(frame_idx);
        var atoms = model.selectedAtoms({});
        cached.push(atoms.map(function(a) { return [a.x, a.y, a.z]; }));
    });
    state.original_coords = cached;
}

function jacobi_eigen_3x3_sym(A_in) {
    // Eigendecomposition of a 3x3 SYMMETRIC matrix via Jacobi
    // rotations. Returns {values: [v1, v2, v3], vectors: V} where
    // V[i][j] = the i-th component of the j-th eigenvector
    // (columns are eigenvectors).
    var A = A_in.map(function(row) { return row.slice(); });
    var V = [[1,0,0],[0,1,0],[0,0,1]];
    for (var iter = 0; iter < 60; iter++) {
        // Find off-diagonal element with largest absolute value.
        var p = 0, q = 1, max_off = Math.abs(A[0][1]);
        if (Math.abs(A[0][2]) > max_off) { p = 0; q = 2; max_off = Math.abs(A[0][2]); }
        if (Math.abs(A[1][2]) > max_off) { p = 1; q = 2; max_off = Math.abs(A[1][2]); }
        if (max_off < 1e-12) break;
        // Compute rotation parameters for Givens.
        var theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
        var t;
        if (Math.abs(theta) > 1e6) {
            t = 0.5 / theta;
        } else {
            t = (theta >= 0 ? 1 : -1)
                / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        }
        var c = 1 / Math.sqrt(t * t + 1);
        var s = c * t;
        var Apq = A[p][q];
        A[p][p] = A[p][p] - t * Apq;
        A[q][q] = A[q][q] + t * Apq;
        A[p][q] = 0;
        A[q][p] = 0;
        for (var r = 0; r < 3; r++) {
            if (r !== p && r !== q) {
                var Arp = A[r][p], Arq = A[r][q];
                A[r][p] = c * Arp - s * Arq;
                A[p][r] = A[r][p];
                A[r][q] = s * Arp + c * Arq;
                A[q][r] = A[r][q];
            }
        }
        for (var r = 0; r < 3; r++) {
            var Vrp = V[r][p], Vrq = V[r][q];
            V[r][p] = c * Vrp - s * Vrq;
            V[r][q] = s * Vrp + c * Vrq;
        }
    }
    return { values: [A[0][0], A[1][1], A[2][2]], vectors: V };
}

function kabsch_rotation(P_centered, Q_centered) {
    // Standard Kabsch derivation:
    //   H = Qᵀ · P  (3×3 cross-covariance)
    //   SVD: H = U Σ Vᵀ
    //   R_optimal = V · diag(1, 1, sign(det(V Uᵀ))) · Uᵀ
    // Returns R such that R · Q_centered[i] ≈ P_centered[i] for all i.
    //
    // We compute V from eigendecomp of HᵀH, then derive U = H · V · Σ⁻¹.
    // Computing U and V independently via two eigendecomps doesn't
    // work because the column orderings + signs come out arbitrary.
    var n = P_centered.length;
    if (n === 0) return [[1,0,0],[0,1,0],[0,0,1]];

    // H[a][b] = (Qᵀ P)[a][b] = Σ_i Q[i][a] · P[i][b]
    var H = [[0,0,0],[0,0,0],[0,0,0]];
    for (var i = 0; i < n; i++) {
        for (var a = 0; a < 3; a++) {
            for (var b = 0; b < 3; b++) {
                H[a][b] += Q_centered[i][a] * P_centered[i][b];
            }
        }
    }

    // HᵀH — 3×3 symmetric. Eigenvectors → V; eigenvalues → σ².
    var HtH = [[0,0,0],[0,0,0],[0,0,0]];
    for (var a = 0; a < 3; a++) {
        for (var b = 0; b < 3; b++) {
            for (var k = 0; k < 3; k++) {
                HtH[a][b] += H[k][a] * H[k][b];
            }
        }
    }
    var eV = jacobi_eigen_3x3_sym(HtH);
    var V = eV.vectors;  // V[i][j] = i-th component of j-th eigenvector
    var sigma = eV.values.map(function(v) { return Math.sqrt(Math.max(v, 0)); });

    // U columns: U_j = (H · V_j) / σ_j. Keeps sign coherent with V.
    // CRITICAL: when σ_j is near zero (rank-deficient H, common when
    // the selected substructure is nearly coplanar / colinear), the
    // division yields garbage and the resulting R is a projection
    // (det near 0), which manifests as the molecule visibly
    // flattening onto a plane. We detect this case and complete the
    // missing U columns via cross products of the good ones, which
    // keeps U orthonormal and R a proper rotation.
    var U = [[0,0,0],[0,0,0],[0,0,0]];
    var sigma_max = Math.max(sigma[0], sigma[1], sigma[2]);
    var deg_threshold = sigma_max * 1e-5 + 1e-10;
    var good_cols = [];
    for (var j = 0; j < 3; j++) {
        if (sigma[j] > deg_threshold) {
            var HVj0 = H[0][0]*V[0][j] + H[0][1]*V[1][j] + H[0][2]*V[2][j];
            var HVj1 = H[1][0]*V[0][j] + H[1][1]*V[1][j] + H[1][2]*V[2][j];
            var HVj2 = H[2][0]*V[0][j] + H[2][1]*V[1][j] + H[2][2]*V[2][j];
            U[0][j] = HVj0 / sigma[j];
            U[1][j] = HVj1 / sigma[j];
            U[2][j] = HVj2 / sigma[j];
            good_cols.push(j);
        }
    }
    complete_U_orthonormal(U, good_cols);

    // R = V · Uᵀ.  Uᵀ[i][j] = U[j][i] → (V·Uᵀ)[i][j] = Σ_k V[i][k] · U[j][k].
    function matmul_VUt(V_, U_) {
        var M = [[0,0,0],[0,0,0],[0,0,0]];
        for (var i = 0; i < 3; i++) {
            for (var j = 0; j < 3; j++) {
                for (var k = 0; k < 3; k++) {
                    M[i][j] += V_[i][k] * U_[j][k];
                }
            }
        }
        return M;
    }
    function det3(M) {
        return M[0][0]*(M[1][1]*M[2][2] - M[1][2]*M[2][1])
             - M[0][1]*(M[1][0]*M[2][2] - M[1][2]*M[2][0])
             + M[0][2]*(M[1][0]*M[2][1] - M[1][1]*M[2][0]);
    }
    var R = matmul_VUt(V, U);
    if (det3(R) < 0) {
        // det(R) = ±1; if negative we have a reflection. Restore a
        // proper rotation by flipping the V column for the smallest σ.
        var idx_min = 0;
        if (sigma[1] < sigma[idx_min]) idx_min = 1;
        if (sigma[2] < sigma[idx_min]) idx_min = 2;
        V[0][idx_min] = -V[0][idx_min];
        V[1][idx_min] = -V[1][idx_min];
        V[2][idx_min] = -V[2][idx_min];
        R = matmul_VUt(V, U);
    }
    return R;
}

function complete_U_orthonormal(U, good_cols) {
    // Given the columns of U that we computed from H · V / σ (the
    // "good" ones, where σ was non-degenerate), fill in the missing
    // columns so U is a proper orthonormal 3×3 matrix.
    //   3 good → already orthonormal (by construction); ensure unit.
    //   2 good → fill the third via cross product of the two.
    //   1 good → pick a perpendicular axis arbitrarily, complete via cross.
    //   0 good → identity (no rotational info; only translation makes sense).
    function colnorm(j) {
        return Math.sqrt(U[0][j]*U[0][j] + U[1][j]*U[1][j] + U[2][j]*U[2][j]);
    }
    function normalize(j) {
        var n = colnorm(j);
        if (n > 1e-12) {
            U[0][j] /= n; U[1][j] /= n; U[2][j] /= n;
        }
    }
    function cross_into(j, a, b) {
        U[0][j] = U[1][a]*U[2][b] - U[2][a]*U[1][b];
        U[1][j] = U[2][a]*U[0][b] - U[0][a]*U[2][b];
        U[2][j] = U[0][a]*U[1][b] - U[1][a]*U[0][b];
    }
    good_cols.forEach(normalize);
    if (good_cols.length === 3) return;
    if (good_cols.length === 2) {
        var a = good_cols[0], b = good_cols[1];
        var missing = [0,1,2].filter(function(x) { return x !== a && x !== b; })[0];
        cross_into(missing, a, b);
        normalize(missing);
        return;
    }
    if (good_cols.length === 1) {
        var a = good_cols[0];
        // Find a unit vector perpendicular to U[:,a]
        var ca = [U[0][a], U[1][a], U[2][a]];
        var seed = Math.abs(ca[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0];
        var dot = seed[0]*ca[0] + seed[1]*ca[1] + seed[2]*ca[2];
        var perp = [seed[0]-dot*ca[0], seed[1]-dot*ca[1], seed[2]-dot*ca[2]];
        var pn = Math.sqrt(perp[0]*perp[0] + perp[1]*perp[1] + perp[2]*perp[2]);
        perp = [perp[0]/pn, perp[1]/pn, perp[2]/pn];
        var others = [0,1,2].filter(function(x) { return x !== a; });
        U[0][others[0]] = perp[0]; U[1][others[0]] = perp[1]; U[2][others[0]] = perp[2];
        cross_into(others[1], a, others[0]);
        normalize(others[1]);
        return;
    }
    // good_cols.length === 0 → identity
    U[0][0] = 1; U[1][0] = 0; U[2][0] = 0;
    U[0][1] = 0; U[1][1] = 1; U[2][1] = 0;
    U[0][2] = 0; U[1][2] = 0; U[2][2] = 1;
}

function compute_kabsch_transform(P, Q) {
    // P = N×3 reference coords, Q = N×3 target coords (to be aligned).
    // Returns {R, t} such that R·Q[i] + t ≈ P[i] for all i in P.
    var n = P.length;
    var pc = [0,0,0], qc = [0,0,0];
    for (var i = 0; i < n; i++) {
        for (var k = 0; k < 3; k++) {
            pc[k] += P[i][k];
            qc[k] += Q[i][k];
        }
    }
    for (var k = 0; k < 3; k++) { pc[k] /= n; qc[k] /= n; }
    var Pc = P.map(function(p) { return [p[0]-pc[0], p[1]-pc[1], p[2]-pc[2]]; });
    var Qc = Q.map(function(q) { return [q[0]-qc[0], q[1]-qc[1], q[2]-qc[2]]; });
    var R = kabsch_rotation(Pc, Qc);
    var Rqc = [
        R[0][0]*qc[0] + R[0][1]*qc[1] + R[0][2]*qc[2],
        R[1][0]*qc[0] + R[1][1]*qc[1] + R[1][2]*qc[2],
        R[2][0]*qc[0] + R[2][1]*qc[1] + R[2][2]*qc[2],
    ];
    return {
        R: R,
        t: [pc[0] - Rqc[0], pc[1] - Rqc[1], pc[2] - Rqc[2]],
    };
}

function apply_transform_inplace(model, R, t) {
    var atoms = model.selectedAtoms({});
    atoms.forEach(function(a) {
        var x = a.x, y = a.y, z = a.z;
        a.x = R[0][0]*x + R[0][1]*y + R[0][2]*z + t[0];
        a.y = R[1][0]*x + R[1][1]*y + R[1][2]*z + t[1];
        a.z = R[2][0]*x + R[2][1]*y + R[2][2]*z + t[2];
    });
}

function set_model_coords(model, coords) {
    // Restore atom positions from a snapshotted N×3 array.
    var atoms = model.selectedAtoms({});
    atoms.forEach(function(a, i) {
        if (!coords[i]) return;
        a.x = coords[i][0];
        a.y = coords[i][1];
        a.z = coords[i][2];
    });
}
