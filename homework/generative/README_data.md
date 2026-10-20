# Homework data: two moons and original MNIST

This assignment deliberately uses the two datasets from the standalone class
flow-matching demos. Students first alter the familiar linear-flow training
design, then learn a new denoising objective and probability-flow ODE on the
same data.

## Synthetic two moons

- **Generator:** [scikit-learn make_moons](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html).
- **Task:** Unconditional generation of 2D points on two noisy crescent-shaped
  branches. The returned binary branch labels may be used for diagnostics but
  are not inputs to the diffusion denoiser.
- **Protocol:** Generate 1,000 training points, 1,000 independent validation
  points, and a second independent 1,000-point evaluation-only real reference
  with noise=0.15, using three distinct recorded random_state values.
  The noise parameter is the standard deviation of the Gaussian perturbation
  added by the generator. Record every seed and any larger reference sample.
- **Features:** Two continuous coordinates; no missing values or learned
  preprocessing. Store training data as float32 tensors of shape (N, 2).
- **Provenance:** The dataset is synthetic, not a record of people or events.
  Cite scikit-learn and the exact generator parameters rather than presenting
  the resulting points as an external observational dataset.

The make_moons random state controls both the point sample and its Gaussian
perturbation. Never reuse the identical seeded realization as both the
optimizer-training and held-out validation reference. For the multi-time
ablation, construct an additional fixed validation set of independent
(x_0, x_1, t) pairs; use it unchanged for all flow designs.

## Original MNIST

- **Source and attribution:** [LeCun, Cortes, and Burges, *The MNIST Database
  of Handwritten Digits*](https://yann.lecun.org/exdb/mnist/). The source
  documents 60,000 official training and 10,000 official test images, each a
  centered 28-by-28 grayscale digit with label 0–9. This is the original MNIST
  dataset, not Fashion-MNIST or the 8-by-8 scikit-learn digits dataset.
- **Loader:** [torchvision.datasets.MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)
  with train=True/False, download=True, and transforms.ToTensor(). The
  loader's root is relative to the directory from which the notebook is
  launched; record the resolved path. The source IDX pixels are unsigned bytes
  in 0–255.
- **Task:** Class-conditional generation of 28-by-28 images for requested
  digits 0–9. Training images and labels are paired; generated images have
  requested labels but no ground-truth class.
- **Split:** Reserve validation rows from the official 60,000-image training
  split before model selection. Use the official test split only for final
  reported checks. A CPU subset must contain at least 6,000
  optimizer-training and 1,000 validation rows sampled with recorded
  class-stratified indices or a fully specified seed and algorithm. Fit a
  separate digit probe only on training images; report its accuracy on
  untouched real test images before using it to score generated images.
- **Preprocessing:** ToTensor() scales source bytes to [0, 1]; set
  x_data = 2 * image - 1 for diffusion training and ODE sampling. Keep
  (batch, 1, 28, 28) tensors. Transform generated values back to [0, 1]
  for displays and pooled-pixel metrics; document any clipping. Do not fit
  normalization, PCA, or evaluator features on the test split.

The original MNIST page is the authoritative dataset description. It does not
provide a simple modern license statement in the linked description, so cite
the data creators and avoid redistributing raw IDX files with the submission.
These digits are derived from real handwritten forms; a nearest-image or
classifier probe is not a privacy guarantee.

## Evaluation protocol

Use a fixed, held-out perturbation set for validation noise MSE. Fresh random
times and Gaussian noise during optimization are appropriate, but the
validation draws should not change from plot point to plot point. Compare
noise MSE with a zero predictor only as an objective baseline.

For two moons, plot generated points with held-out real points and compute
RBF MMD squared with a bandwidth chosen only from real validation data and
then frozen for every model. State whether the MMD estimator includes
diagonal kernel terms. A real-versus-real comparison should use the same
sample counts as generated-versus-real: compare the validation set with the
disjoint evaluation-only set. It provides a finite-sample reference, not a zero
target.

For MNIST, use a fixed 14-by-14 average-pooled [0, 1] pixel representation
for the required RBF MMD squared; choose its bandwidth on held-out real
images and reuse it. Match the generated and real groups by count per digit;
use two disjoint class-matched real groups for the real-versus-real reference.
Report all sample counts. A separate digit probe gives conditional label
agreement; its measured accuracy on official real test images is needed to
interpret that number. Neither pooled-pixel MMD nor probe agreement is
Inception-feature FID, human assessment, or evidence that images are not
memorized. Include uncurated sample grids and visibly imperfect samples.

Record the model seed, data seed, time/noise seed, sampler seed, initial
Gaussian draws, Euler or RK2 step count, and number of network evaluations.
For Problem 1, also record both optimizer-update count and processed
(pair, time) examples, since the multi-time designs consume different
budgets under update matching.

## Source readings

- [Lipman et al., *Flow Matching Guide and Code*](https://arxiv.org/abs/2412.06264):
  recommended background for conditional paths and flow matching.
- [Song et al., *Score-Based Generative Modeling through Stochastic
  Differential Equations*](https://arxiv.org/abs/2011.13456): optional advanced
  background on diffusion SDEs and probability-flow ODEs. The homework supplies
  every formula needed for implementation; no SDE or score derivation is required.
- [scikit-learn make_moons documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html)
  and [official MNIST dataset description](https://yann.lecun.org/exdb/mnist/):
  data provenance.
