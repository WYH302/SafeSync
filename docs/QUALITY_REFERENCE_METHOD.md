# Receiver-relative quality decisions from compact references

Status: implemented research candidate, 2026-10-28. This replaces the claim that
renaming ACK deduplication as `unknown × evidence deficit` is an innovation.
The underlying projection and concentration tools are established techniques.
Publication readiness depends on the measured codec/cache/risk trade-off and
the remaining direct comparisons, not on a new name for these tools.

## Research question and information contract

The ground receiver stores a decoded historical RGB mosaic. The satellite has
a limited persistent reference budget for each geographic unit. Can a short
reference support useful decisions about which historical detail may be reused,
while detecting excessive reconstruction error independently of semantic class?

The ground sends a compact reference computed from its **actual reconstructed
state**. This reference is bound to the acknowledged state version/digest and
the expected observation number. A fresh Gaussian matrix is generated for each
observation; its seed is transmitted in the reference. In the current experiment
the ground regenerates the reference every window. Its complete byte length is
uplink traffic. This assumption is material: the ground has storage and compute,
and the link can provide a reference before the next downlink decision.

The sender sees current RGB, the bound reference, and acknowledged metadata.
Semantic labels, cloud GT, current ground-truth masks, and the ground's historical
RGB do not enter its compact-reference decisions. A larger-cache baseline is
explicitly allowed to retain the full historical RGB and uses the same codec.
It has a different persistent-memory cost and is not called a GT oracle.

## Common coarse update

For each 64×64 RGB block, let `D` take means over non-overlapping 16×16 cells and
let `U` replicate these means. `UD` is an orthogonal projection in Euclidean
pixel space. Define `h(x)=x-UDx`, using RGB in [0,1].

Each transmitted observation includes the quantized current means `Q8(Dx)` in
one PNG image. Before receiving any fine ROI, the ground reconstructs

\[
 z=\operatorname{round}_{8}\!\left[\operatorname{clip}_{[0,1]}
   \{y-UDy+UQ_8(Dx)\}\right],
\]

where `y` is the actual prior decoded mosaic. Selected 64×64 regions then replace
the corresponding pixels with an actual decoded JPEG atlas. This is an explicit
low-frequency update, not physical-world silence. Manifest value `STABLE` in the
prototype means **historical detail accepted for reuse after this coarse update**.
Other manifest outcomes are full ROI received and fine update deferred.

Coarse PNG, ROI indices, image headers, manifests, digests, transactions,
initialization, ACKs, and reference uploads are counted. No padding is sent to
manufacture equal rates. All receiver task evaluation must replay these packets.

## Pixel-MSE bound (standard Gaussian norm estimation)

Flatten `d=h(x)-h(y)` into `D_b=64×64×3` components. Let the rows of `A` be
independent `N(0,I/D_b)`. For `k` rows, let `q=Q16(Ah(y))`, with quantization step
`Δ=1/256`, and compute `v=Ah(x)-q`. For fixed `x,y` independent of this matrix,

\[
 \frac{\|Ad\|_2^2}{\|d\|_2^2/D_b}\sim\chi_k^2,
 \qquad \|Ad\|_2\le\|v\|_2+\sqrt{k}\,\Delta/2.
\]

With `c=χ²_k(α)` the lower-tail α quantile,

\[
 H=\frac{(\|v\|_2+\sqrt{k}\Delta/2)^2}{c}
\]

satisfies `P_A(||d||²/D_b > H) ≤ α`. Current coarse quantization error
`q_c=||Dx-Q8(Dx)||² / dim(Dx)` is known exactly at the sender. Orthogonality,
nonexpansiveness of clipping, and final 8-bit rounding give

\[
 \operatorname{MSE}(x,z)
 \le (\sqrt{H+q_c}+1/(2\cdot255))^2 = B
\]

with the same probability. A block is accepted for reuse only if `B≤τ`, where
`τ=10^{-PSNR/10}`. Overflows cause rejection/fallback, never silent saturation.
The argument is over real arithmetic and ideal independent Gaussian rows; the
implementation uses float32 PRNG/matrix arithmetic and is checked numerically.
It is not a formally verified floating-point bound or an adversarial guarantee.

The prototype initially evaluated separate RGB sketches with α/3 union bounds.
The joint RGB sketch uses the same number of serialized coefficient bytes and
avoids that unnecessary conservatism. Coarse-plus-projection and DCT variants
remain development comparisons, not additional claimed inventions.

## What the probability means

For a fixed **bad** current/reference pair with true MSE above τ, the probability
over a fresh random sketch of accepting that pair is at most α. This is not a
bound on the fraction of errors **among accepted blocks**, nor a simultaneous
whole-mission guarantee. Both denominators are reported separately.

The physical observations must be independent of the fresh random matrix. Prior
receiver state may depend on earlier sketches. Reusing the same matrix after
sketch-dependent state selection does not meet the stated premise. Predetermined
pseudorandom seeds are an experimental implementation of the fresh randomness
assumption; malicious adaptive observations are outside the claim.

The bound concerns whole-block RGB MSE. A small unknown object can still have
large local error while the block MSE is low. It provides no guarantee about
arbitrary semantic labels, thin structures, future tasks, registration accuracy,
or physical-world absence of change. Object/unknown-region quality and receiver
tasks must therefore be evaluated separately. Unsupported semantic guarantees
must not appear in the manuscript.

## Strong comparisons and protocol

1. Raw thumbnail error; a 95th-percentile multiplicative correction fitted only
   on the two original calibration AOIs. Correlated block calibration has no IID
   coverage claim. The side-4 thumbnail is only an ablation after coarse replacement:
   it contains no additional detail. It cannot carry the main comparison.
2. DCT coefficients plus upward-rounded residual norms, a deterministic
   triangle-inequality bound at the same reference payload budget.
3. JPEG/JPEG2000 compact reference plus upward-rounded error norms, using a
   server-selected resolution/quality under an exact reference byte cap. This
   protects a strong mature-codec alternative to the random summary.
4. Full historical RGB cache with exact current/coarse-restored MSE; same ROI
   format and budget. Report its actual cache size rather than hide its input.
5. Full current JPEG and JPEG2000 fitted to the same actual downlink caps.
   These may win; historical reuse is not presumed superior.

The receiver validates region, base version/digest, observation order, length,
manifest/payload agreement, and checksum. A transaction remains pending until
its matching counted ACK. Duplicate last transactions return the same ACK and
are not applied twice. New observations cannot overtake pending transactions.
The current main link has immediate ACK and no erasure. Fault handling tests
are protocol evidence only, not real-radio or production-security validation.

## Experimental scope and separation

The original four SpaceNet7 AOIs are development data: two for calibration and
two for development evaluation, all selected using geography before imagery.
There are 96 genuine monthly mosaics, with actual missing months and UDM masks.
Statistics aggregate by AOI; months, tiles, and pixels are not independent scenes.
The 192 new SECOND pairs are 128 calibration and 64 validation pairs. Their
official test images remain unread. The newly reserved SpaceNet7 AOI prefix is
also not available for method selection; confirmation starts after code/config
freeze. The full current information/old-state ground-evaluator checks precede
any task-performance claims.

## Literature boundaries

Earth+ (ASPLOS 2025; https://arxiv.org/html/2403.11434v1) already uses ground
history, compact references, changed regions, and mature codecs. FOOL
(https://arxiv.org/html/2403.16677v3) already motivates task-agnostic information
retention. Anomaly-Based JPEG2000 Compression of Hyperspectral Imagery (Du,
Zhu, Fowler, GRSL 2008, DOI 10.1109/LGRS.2008.2003180) already protects anomaly
pixels. Reduced-reference IQA and random-projection macroblock MSE estimation
also predate this work (e.g., Reduced-Reference Video Quality Assessment using
Distributed Source Coding; original indexed preprint located, full access still
being resolved). None of these generic ideas can be claimed as first use here.
Any contribution must be the specific prospective reuse decision, actual
closed-loop resource trade-off, and evidence demonstrated by the experiments.
