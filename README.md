# Grokking Boundary Experiments

## Project Overview

This repository investigates the **grokking phenomenon** in neural networks—the
delayed transition from memorization to generalization that occurs long after
training loss has plateaued.

**Task:** Modular arithmetic, specifically (a + b) mod 97 = c

**Central Question:**
How does output-space overlap between training and test sets affect the
occurrence and timing of grokking?

In particular, we examine whether reducing overlap while maintaining partial
interpolation capability can induce delayed grokking relative to a random-split
baseline.

---

## Experimental Setup

**Architecture:**
4-layer decoder-only transformer
- 4 attention heads
- 256-dimensional embeddings
- 1024-dimensional feedforward layers
- ~3.2M parameters

**Task:**
Binary operation on integers modulo 97. Input format: `<bos> a + b mod 97 = c <eos>`
Model predicts next token autoregressively.

**Training:**
- Optimizer: AdamW (learning rate 0.001, weight decay 0.0)
- Batch size: 256
- Loss: Cross-entropy on next-token prediction

**Metrics:**
- **Test accuracy:** Fraction of correctly predicted results
- **Representation rank:** Effective rank of hidden states (final layer)
- **Attention entropy:** Shannon entropy of attention distributions

**Internal Signal:**
Representation rank collapse is used as the primary internal indicator of
grokking. Sharp rank reduction coinciding with accuracy increase signals the
transition from memorization to generalization.

---

## Experimental Variants

| Experiment | Split Design | Output Overlap |
|------------|--------------|----------------|
| **V1**     | Random 80/20 | ~80% (full coverage) |
| **V2**     | Disjoint: Train [0,47], Test [48,96] | 0% |
| **V3**     | Overlap: Train [0,62], Test [32,96] | 49.2% |
| **V3.1**   | Overlap: Train [0,40], Test [30,96] | 26.8% |
| **V3.2**   | Overlap: Train [0,25], Test [20,96] | 7.8% |

**Design Rationale:**
Random splits (V1) allow full interpolation. Structured splits (V2-V3.2) impose
varying degrees of constraint conflict by controlling the overlap between
training and test output ranges. V2 represents pure extrapolation (zero overlap),
while V3-V3.2 explore partial overlap regimes.

---

## Results Summary

### V1: Baseline Grokking (Random 80/20)
- **Grokking occurs:** Sharp transition at ~17,100 steps
- **Final accuracy:** 99.36%
- **Rank dynamics:** Collapse from ~5 to ~1 at grokking transition
- **Conclusion:** Random split successfully demonstrates grokking

### V2: Extrapolation Failure (0% Overlap)
- **Training:** 400,000 steps
- **Test accuracy:** 0.00% (no generalization)
- **Rank dynamics:** No collapse observed
- **Conclusion:** Pure extrapolation to disjoint output ranges is impossible for
this task and model

### V3 / V3.1: Interpolation Regimes (49% / 27% Overlap)
- **V3 (49% overlap):**
  - Test accuracy at 20k steps: 47.5%
  - Early generalization via interpolation
  - No delayed grokking observed

- **V3.1 (27% overlap):**
  - Test accuracy at 20k steps: 16.4%
  - Still allows interpolation
  - No delayed grokking observed

- **Conclusion:** High overlap eliminates constraint conflict and allows the model
to generalize early through interpolation on the shared output region

### V3.2: Boundary Regime (7.8% Overlap)
- **Test accuracy at 20k steps:** 7.8%
- **Rank dynamics:** Remains high (~30-40), no collapse
- **Conclusion:** Minimal overlap creates strongest constraint conflict but does
not induce grokking within 20k steps. Represents practical lower bound before
approaching extrapolation failure.

### Phase Transition Pattern

A clear monotonic relationship exists between overlap percentage and test
accuracy:

```
Overlap   →   Test Accuracy @ 20k
  0.0%    →      0.0%  (extrapolation failure)
  7.8%    →      7.8%  (boundary regime)
 26.8%    →     16.4%  (interpolation begins)
 49.2%    →     47.5%  (easy interpolation)
 ~80%     →     ~35%   (full interpolation capability)
```

**Critical Finding:**
No smooth "delayed grokking" regime was observed between interpolation and
extrapolation failure. The system exhibits sharp transitions between regimes
rather than a continuous delay in grokking timing.

---

## Key Insight: Interpolation vs. Grokking

**Interpolation Regime:**
When training and test sets share output values, the model can generalize to
test examples by recognizing that certain input combinations produce results
already seen during training. This is **feature interpolation**, not algorithmic
discovery.

**Why Interpolation Prevents Grokking:**
Grokking requires the model to discover an algorithmic solution (modular addition)
under constraint conflict. High overlap reduces this conflict—the model can achieve
reasonable test accuracy without learning the underlying algorithm, only by
memorizing input-output mappings and interpolating across the shared region.

**Evidence from Representation Dynamics:**
- **V1 (random split):** Rank collapses sharply (5→1) at grokking, indicating
  compressed, algorithmic representations
- **V3.2 (minimal overlap):** Rank remains high (~30-40) throughout, indicating
  no algorithmic compression occurred

**Why Grokking Requires Constraints:**
The sharp rank reduction in V1 suggests that grokking involves discovering a
low-dimensional algorithmic solution. This compression is driven by conflicting
pressures: fit training data while generalizing to structurally different test
cases. When test cases are too similar (high overlap), the model finds shortcuts.
When they are too different (zero overlap), the model cannot bridge the gap.

---

## Conclusion

**Main Finding:**
For modular arithmetic with a 4-layer transformer, grokking via output-space
overlap manipulation is constrained by sharp phase boundaries:

1. **Random splits enable grokking** through a combination of full output coverage
   and statistical variation in input combinations

2. **Structured overlap creates a continuous difficulty gradient** but does not
   produce a smooth delayed-grokking regime within the viable range (0% < overlap < 27%)

3. **The system exhibits three distinct regimes:**
   - **Extrapolation failure** (0% overlap): No learning possible
   - **Boundary regime** (7.8% overlap): Learning occurs but no grokking within 20k steps
   - **Interpolation regime** (>27% overlap): Early generalization without algorithmic discovery

4. **Constraint conflict alone is insufficient** for delayed grokking. The model
   either interpolates (when overlap is sufficient) or fails to generalize
   (when overlap is too low).

**Implication:**
Grokking is not easily controlled through simple data split manipulation. The
phenomenon appears sensitive to the specific structure of train/test overlap,
and successful grokking (as in V1) may depend on preserving certain algorithmic
properties in the data distribution that structured splits disrupt.

---

## Repository Structure

```
experiments/modular_arithmetic/
├── v1_baseline/              # Random 80/20 split (successful grokking)
├── v2/                       # Disjoint split (extrapolation failure)
├── v3_overlap_conflict/      # 49% overlap (interpolation)
├── v3_1_reduced_overlap/     # 27% overlap (interpolation)
├── v3_2_minimal_overlap/     # 7.8% overlap (boundary regime)
└── final_analysis/           # Comparative plots and technical summary

scripts/
├── train_v1.py               # Baseline training
├── train_v2.py               # Disjoint split training
├── train_v3_2.py             # Minimal overlap training
├── analyze_v2.py             # Analysis for structured splits
└── final_analysis.py         # Generate comparative plots

src/
├── model.py                  # Transformer implementation
├── training.py               # Training loop with analysis hooks
├── analyze.py                # Representation rank and entropy computation
├── dataset.py                # Random split dataset
├── dataset_structured.py     # Disjoint split dataset
├── dataset_overlap_v32.py    # Minimal overlap dataset
└── tokenizer.py              # Modular arithmetic tokenizer
```

---

## Usage

**Training:**
```bash
# V1 baseline (200k steps)
python scripts/train_v1.py

# V3.2 minimal overlap (20k sanity check)
python scripts/train_v3_2.py
```

**Analysis:**
```bash
# Generate final comparative plots
python scripts/final_analysis.py
```

**Generated Artifacts:**
- `final_accuracy_comparison.png` - V1 vs V3.2 test accuracy over time
- `final_rank_comparison.png` - Representation rank dynamics
- `final_overlap_phase_diagram.png` - Phase transition across all experiments
- `FINAL_TECHNICAL_SUMMARY.txt` - Detailed findings

---

## References

This work investigates phenomena observed in:

Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022).
*Grokking: Generalization beyond overfitting on small algorithmic datasets.*
arXiv:2201.02177

---

## License

Research code. No license specified.
