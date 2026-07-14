//! Scratch-bounding capacity constants. `glmm` owns these; they
//! size the solver's stack buffers, so they are HARD ceilings. These ceilings
//! live here in `consts` as the single source, no drift.

/// Max primary RE width `q_p = 1 + #slopes` (≤ 7 random slopes). This
/// is the NoZ-scratch envelope, not a model ceiling — over-envelope designs
/// route to the sparse-Z path (`fit::classify_design`).
pub const MAX_PRIMARY_Q: usize = 8;

/// Max extra grouping factors. Mirrors the solver's per-row level-id buffer
/// width `1 + MAX_EXTRA_GROUPINGS`. 6 keeps the NoZ envelope's total variance
/// components `MAX_PRIMARY_Q + MAX_EXTRA_GROUPINGS·MAX_EXTRA_Q = 8 + 6·4 = 32`
/// well within the `pinned_components` bitmask. This is the
/// NoZ-scratch envelope, not a model ceiling — over-envelope designs route to
/// the sparse-Z path (`fit::classify_design`). The mask is now `u64`, so the
/// sparse path safely pins up to 64 components; raising MAX_* past 64 total
/// components would require replacing the bitmask with a bitset.
pub const MAX_EXTRA_GROUPINGS: usize = 6;

/// Max random-effect dimension per EXTRA grouping factor: `q_g = 1 + #slopes`
/// (intercept + up to 3 random slopes). Bounds the per-factor `vech(Λ_g)` θ block
/// and every stack buffer sized off `MAX_THETA`. 4 covers the realistic ceiling
/// (a crossed factor with intercept + 3 slopes); raise only with a benchmark. This
/// is the NoZ-scratch envelope, not a model ceiling — over-envelope
/// designs route to the sparse-Z path (`fit::classify_design`).
pub const MAX_EXTRA_Q: usize = 4;

/// Max θ length: full primary `vech(Λ_p)` plus a `vech(Λ_g)` block per extra
/// grouping. For `MAX_EXTRA_Q = 1` this collapses to one scalar per extra — the
/// pre-slope bound `… + MAX_EXTRA_GROUPINGS`.
pub const MAX_THETA: usize = MAX_PRIMARY_Q * (MAX_PRIMARY_Q + 1) / 2
    + MAX_EXTRA_GROUPINGS * (MAX_EXTRA_Q * (MAX_EXTRA_Q + 1) / 2);

/// Max total level count over `Crossed` extra groupings on the dense NoZ path:
/// designs with `Σ n_clusters` over crossed extras above this route to the
/// sparse-Z path (`fit::classify_design`). Unlike the `MAX_*` scratch ceilings
/// this is a PERFORMANCE boundary — the dense tail of `reml_deviance` (and the
/// dense GLMM path's Schur complement over extras) is cubic in the total
/// crossed column count, so many-level crossed factors make each deviance eval
/// take minutes (measured 2026-07-09: 22,714 crossed levels ≈ 10¹³ flops and
/// ~6 GB scratch per eval). 500 is a placeholder: grouseticks' 403-level
/// crossed factor ran fine dense (0.16 s), and the cubic term at 500 is ~4×10⁷
/// flops/eval — negligible. Refine with a dense-vs-sparse crossover sweep
/// extending the 2026-07-02 sweep's level axis past 30.
pub const MAX_CROSSED_LEVELS: usize = 500;

/// Max adaptive Gauss–Hermite node count for `nAGQ>1`. Odd nodes
/// only; the GH table below covers orders `1,3,…,MAX_NAGQ`. Uniform across the
/// scalar and vector (`agq::agq_deviance_vec`, q_p≤3) AGQ paths: the vector path
/// uses a `k^q` **product** grid, so k=25 at q=3 is 15,625 nodes/cluster/eval —
/// legal but self-punishing; the k^q cost is the user's to pay (the q_p≤3 cap in
/// `assert_model_shape` bounds the exponent).
pub const MAX_NAGQ: u8 = 25;

// --- Gauss–Hermite quadrature table (physicists', weight e^{-x²}) ------------
// Flattened nodes/weights for the odd orders 1,3,5,…,25, concatenated in order.
// The block for order k = 2i+1 occupies `GH_NODES[GH_OFFSETS[i]..GH_OFFSETS[i+1]]`
// (offsets are the perfect squares i²). Used by the AGQ deviance: a per-eval
// const slice, zero allocation. Generated via Golub–Welsch (numpy
// `hermgauss`); each order's weights sum to √π (verified at generation,
// re-checked in `gh_table_weights_sum_sqrt_pi`). To map nAGQ=k → block index use
// `i = (k-1)/2`.
/// Block offsets into [`GH_NODES`]/[`GH_WEIGHTS`]: order `k=2i+1` occupies
/// `[GH_OFFSETS[i]..GH_OFFSETS[i+1]]`. Entries are the perfect squares `i²`
/// (order `2i+1` contributes `2i+1` nodes; the running sum of odds is `i²`).
pub const GH_OFFSETS: [usize; 14] = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81, 100, 121, 144, 169];

/// GH abscissae `z_j` (physicists'), concatenated over orders 1,3,…,25. Indexed
/// via [`GH_OFFSETS`]. The adaptive node is `u_cj = ũ_c + √2·σ_c·z_j`.
// Literals carry the generator's full digits for provenance; the 18th figure is
// past f64 precision (rounds to the same bits), so the cosmetic excessive_precision
// lint is allowed rather than truncating the verified table.
#[allow(clippy::excessive_precision)]
pub const GH_NODES: [f64; 169] = [
    0.00000000000000000e+00,
    -1.22474487139158894e+00,
    0.00000000000000000e+00,
    1.22474487139158894e+00,
    -2.02018287045608558e+00,
    -9.58572464613818509e-01,
    0.00000000000000000e+00,
    9.58572464613818509e-01,
    2.02018287045608558e+00,
    -2.65196135683523337e+00,
    -1.67355162876747143e+00,
    -8.16287882858964586e-01,
    0.00000000000000000e+00,
    8.16287882858964586e-01,
    1.67355162876747143e+00,
    2.65196135683523337e+00,
    -3.19099320178152768e+00,
    -2.26658058453184319e+00,
    -1.46855328921666795e+00,
    -7.23551018752837560e-01,
    0.00000000000000000e+00,
    7.23551018752837560e-01,
    1.46855328921666795e+00,
    2.26658058453184319e+00,
    3.19099320178152768e+00,
    -3.66847084655958255e+00,
    -2.78329009978165143e+00,
    -2.02594801582575545e+00,
    -1.32655708449493281e+00,
    -6.56809566882099793e-01,
    0.00000000000000000e+00,
    6.56809566882099793e-01,
    1.32655708449493281e+00,
    2.02594801582575545e+00,
    2.78329009978165143e+00,
    3.66847084655958255e+00,
    -4.10133759617863980e+00,
    -3.24660897837240991e+00,
    -2.51973568567823758e+00,
    -1.85310765160151192e+00,
    -1.22005503659074832e+00,
    -6.05763879171060116e-01,
    0.00000000000000000e+00,
    6.05763879171060116e-01,
    1.22005503659074832e+00,
    1.85310765160151192e+00,
    2.51973568567823758e+00,
    3.24660897837240991e+00,
    4.10133759617863980e+00,
    -4.49999070730939188e+00,
    -3.66995037340445274e+00,
    -2.96716692790560321e+00,
    -2.32573248617385797e+00,
    -1.71999257518648885e+00,
    -1.13611558521092060e+00,
    -5.65069583255575769e-01,
    0.00000000000000000e+00,
    5.65069583255575769e-01,
    1.13611558521092060e+00,
    1.71999257518648885e+00,
    2.32573248617385797e+00,
    2.96716692790560321e+00,
    3.66995037340445274e+00,
    4.49999070730939188e+00,
    -4.87134519367440344e+00,
    -4.06194667587547453e+00,
    -3.37893209114149418e+00,
    -2.75776291570388876e+00,
    -2.17350282666662054e+00,
    -1.61292431422123128e+00,
    -1.06764872574345060e+00,
    -5.31633001342654787e-01,
    0.00000000000000000e+00,
    5.31633001342654787e-01,
    1.06764872574345060e+00,
    1.61292431422123128e+00,
    2.17350282666662054e+00,
    2.75776291570388876e+00,
    3.37893209114149418e+00,
    4.06194667587547453e+00,
    4.87134519367440344e+00,
    -5.22027169053748175e+00,
    -4.42853280660377902e+00,
    -3.76218735196402010e+00,
    -3.15784881834760212e+00,
    -2.59113378979454279e+00,
    -2.04923170985061898e+00,
    -1.52417061939353315e+00,
    -1.01036838713431143e+00,
    -5.03520163423888167e-01,
    0.00000000000000000e+00,
    5.03520163423888167e-01,
    1.01036838713431143e+00,
    1.52417061939353315e+00,
    2.04923170985061898e+00,
    2.59113378979454279e+00,
    3.15784881834760212e+00,
    3.76218735196402010e+00,
    4.42853280660377902e+00,
    5.22027169053748175e+00,
    -5.55035187326467838e+00,
    -4.77399234341121925e+00,
    -4.12199554749184038e+00,
    -3.53197287713767771e+00,
    -2.97999120770459802e+00,
    -2.45355212451283800e+00,
    -1.94496294918625368e+00,
    -1.44893425065073189e+00,
    -9.61499634418369054e-01,
    -4.79450707079107530e-01,
    0.00000000000000000e+00,
    4.79450707079107530e-01,
    9.61499634418369054e-01,
    1.44893425065073189e+00,
    1.94496294918625368e+00,
    2.45355212451283800e+00,
    2.97999120770459802e+00,
    3.53197287713767771e+00,
    4.12199554749184038e+00,
    4.77399234341121925e+00,
    5.55035187326467838e+00,
    -5.86430949898457232e+00,
    -5.10153461047667722e+00,
    -4.46209117374000641e+00,
    -3.88447270810610190e+00,
    -3.34512715994122445e+00,
    -2.83180378712615699e+00,
    -2.33701621147445593e+00,
    -1.85567703767137093e+00,
    -1.38403958568249519e+00,
    -9.19151465442563764e-01,
    -4.58538350068104783e-01,
    0.00000000000000000e+00,
    4.58538350068104783e-01,
    9.19151465442563764e-01,
    1.38403958568249519e+00,
    1.85567703767137093e+00,
    2.33701621147445593e+00,
    2.83180378712615699e+00,
    3.34512715994122445e+00,
    3.88447270810610190e+00,
    4.46209117374000641e+00,
    5.10153461047667722e+00,
    5.86430949898457232e+00,
    -6.16427243405245218e+00,
    -5.41363635528003329e+00,
    -4.78532036735222377e+00,
    -4.21860944438656116e+00,
    -3.69028287699835600e+00,
    -3.18829492442510487e+00,
    -2.70532023717302605e+00,
    -2.23642013026728081e+00,
    -1.77800112433714741e+00,
    -1.32728070207308391e+00,
    -8.81982756213821273e-01,
    -4.40147298645308327e-01,
    0.00000000000000000e+00,
    4.40147298645308327e-01,
    8.81982756213821273e-01,
    1.32728070207308391e+00,
    1.77800112433714741e+00,
    2.23642013026728081e+00,
    2.70532023717302605e+00,
    3.18829492442510487e+00,
    3.69028287699835600e+00,
    4.21860944438656116e+00,
    4.78532036735222377e+00,
    5.41363635528003329e+00,
    6.16427243405245218e+00,
];

/// GH weights `w_j` (physicists'), aligned 1:1 with [`GH_NODES`]. Per order they
/// sum to √π; the AGQ log-sum-exp uses `ln w_j` and the `√π` normalization is
/// what makes the k=1 node collapse exactly to the Laplace term.
#[allow(clippy::excessive_precision)]
pub const GH_WEIGHTS: [f64; 169] = [
    1.77245385090551588e+00,
    2.95408975150919406e-01,
    1.18163590060367718e+00,
    2.95408975150919406e-01,
    1.99532420590459170e-02,
    3.93619323152241074e-01,
    9.45308720482941789e-01,
    3.93619323152241074e-01,
    1.99532420590459170e-02,
    9.71781245099519902e-04,
    5.45155828191270508e-02,
    4.25607252610127773e-01,
    8.10264617556807232e-01,
    4.25607252610127773e-01,
    5.45155828191270508e-02,
    9.71781245099519902e-04,
    3.96069772632643647e-05,
    4.94362427553694112e-03,
    8.84745273943766397e-02,
    4.32651559002555641e-01,
    7.20235215606050971e-01,
    4.32651559002555641e-01,
    8.84745273943766397e-02,
    4.94362427553694112e-03,
    3.96069772632643647e-05,
    1.43956039371425964e-06,
    3.46819466323345445e-04,
    1.19113954449115069e-02,
    1.17227875167708509e-01,
    4.29359752356124946e-01,
    6.54759286914591621e-01,
    4.29359752356124946e-01,
    1.17227875167708509e-01,
    1.19113954449115069e-02,
    3.46819466323345445e-04,
    1.43956039371425964e-06,
    4.82573185007312514e-08,
    2.04303604027070872e-05,
    1.20745999271938600e-03,
    2.08627752961699532e-02,
    1.40323320687023495e-01,
    4.21616296898543186e-01,
    6.04393187921161257e-01,
    4.21616296898543186e-01,
    1.40323320687023495e-01,
    2.08627752961699532e-02,
    1.20745999271938600e-03,
    2.04303604027070872e-05,
    4.82573185007312514e-08,
    1.52247580425352091e-09,
    1.05911554771106247e-06,
    1.00004441232499824e-04,
    2.77806884291277503e-03,
    3.07800338725460997e-02,
    1.58488915795935714e-01,
    4.12028687498898705e-01,
    5.64100308726417365e-01,
    4.12028687498898705e-01,
    1.58488915795935714e-01,
    3.07800338725460997e-02,
    2.77806884291277503e-03,
    1.00004441232499824e-04,
    1.05911554771106247e-06,
    1.52247580425352091e-09,
    4.58057893079860965e-11,
    4.97707898163076990e-08,
    7.11228914002129320e-06,
    2.98643286697753123e-04,
    5.06734995762754461e-03,
    4.09200341497562917e-02,
    1.72648297670096984e-01,
    4.01826469470412062e-01,
    5.30917937624863501e-01,
    4.01826469470412062e-01,
    1.72648297670096984e-01,
    4.09200341497562917e-02,
    5.06734995762754461e-03,
    2.98643286697753123e-04,
    7.11228914002129320e-06,
    4.97707898163076990e-08,
    4.58057893079860965e-11,
    1.32629709449852338e-12,
    2.16305100986357480e-09,
    4.48824314722311525e-07,
    2.72091977631617121e-05,
    6.70877521407180406e-04,
    7.98886677772302559e-03,
    5.08103869090520063e-02,
    1.83632701306997104e-01,
    3.91608988613030284e-01,
    5.02974888276186527e-01,
    3.91608988613030284e-01,
    1.83632701306997104e-01,
    5.08103869090520063e-02,
    7.98886677772302559e-03,
    6.70877521407180406e-04,
    2.72091977631617121e-05,
    4.48824314722311525e-07,
    2.16305100986357480e-09,
    1.32629709449852338e-12,
    3.72036507013602274e-14,
    8.81861124204993316e-11,
    2.57123018005931538e-08,
    2.17188489805666986e-06,
    7.47839886731006283e-05,
    1.25498204172640876e-03,
    1.14140658374343971e-02,
    6.01796466589123030e-02,
    1.92120324066997750e-01,
    3.81669073613502219e-01,
    4.79023703120177557e-01,
    3.81669073613502219e-01,
    1.92120324066997750e-01,
    6.01796466589123030e-02,
    1.14140658374343971e-02,
    1.25498204172640876e-03,
    7.47839886731006283e-05,
    2.17188489805666986e-06,
    2.57123018005931538e-08,
    8.81861124204993316e-11,
    3.72036507013602274e-14,
    1.01603846206368036e-15,
    3.40831409803052011e-12,
    1.35962965040289660e-09,
    1.55533932914576691e-07,
    7.24929591800226328e-06,
    1.65561699141874273e-04,
    2.06956787496063912e-03,
    1.52070840044841397e-02,
    6.88902894290874257e-02,
    1.98644898578022450e-01,
    3.72143824877564977e-01,
    4.58196585593213190e-01,
    3.72143824877564977e-01,
    1.98644898578022450e-01,
    6.88902894290874257e-02,
    1.52070840044841397e-02,
    2.06956787496063912e-03,
    1.65561699141874273e-04,
    7.24929591800226328e-06,
    1.55533932914576691e-07,
    1.35962965040289660e-09,
    3.40831409803052011e-12,
    1.01603846206368036e-15,
    2.71192351403839545e-17,
    1.25881498774654906e-13,
    6.71963841770625252e-11,
    1.01703825030184802e-08,
    6.25703249969110755e-07,
    1.89159729573404554e-05,
    3.15083638745484150e-04,
    3.11570872012563015e-03,
    1.92430989654088953e-02,
    7.68889951758088691e-02,
    2.03621136678124037e-01,
    3.63088989275890450e-01,
    4.39868722169484971e-01,
    3.63088989275890450e-01,
    2.03621136678124037e-01,
    7.68889951758088691e-02,
    1.92430989654088953e-02,
    3.11570872012563015e-03,
    3.15083638745484150e-04,
    1.89159729573404554e-05,
    6.25703249969110755e-07,
    1.01703825030184802e-08,
    6.71963841770625252e-11,
    1.25881498774654906e-13,
    2.71192351403839545e-17,
];

#[cfg(test)]
mod tests {
    use super::*;

    /// Per-order GH weights sum to √π (the table's defining normalization; the
    /// AGQ k=1=Laplace reduction depends on it). Cross-checks the pasted literals
    /// against the closed-form sum rather than trusting the generator.
    #[test]
    fn gh_table_weights_sum_sqrt_pi() {
        let sqrt_pi = std::f64::consts::PI.sqrt();
        for i in 0..GH_OFFSETS.len() - 1 {
            let block = &GH_WEIGHTS[GH_OFFSETS[i]..GH_OFFSETS[i + 1]];
            let s: f64 = block.iter().sum();
            assert!(
                (s - sqrt_pi).abs() < 1e-13,
                "order {} weights sum {} != √π",
                2 * i + 1,
                s
            );
        }
        assert_eq!(GH_OFFSETS[GH_OFFSETS.len() - 1], GH_NODES.len());
    }
}
