# Support router

The teacher chooses which columns train on each task. At inference, the **support router** selects one of the saved sets of nonshared columns from the input alone. The runner adds the always-active shared columns. The router is a reusable component for MNIST grayscale and CIFAR RGB inputs. **Only the Split-MNIST backprop runner currently uses it.**

## Shared component

A `RouterProfile` defines how to encode an input, which small changes to allow when matching it to a centroid (its tangent directions and bounds), how many centroids to store per support, and the fitting seed sequence. The encoder and tangent basis must use the same feature coordinates. For a CIFAR pixel profile, the existing `image_tangents` helper accepts a 32×32×3 shape and per-channel normalized background values. Choosing the CIFAR profile and attaching it to a training boundary are future integration decisions. [Router and profile API](../hibacaml/control/router.py) · [CIFAR preprocessing](../hibacaml/data/cifar.py)

`Router.acquire()` fits k-means centroids using only the current task’s fit inputs and associates them with its final saved support. Earlier entries stay frozen. `Router.route()` scores each eligible support by its closest centroid under bounded tangent distance. It returns the support catalogue, per-support scores, selected indices, and the margin between the best two supports when a choice exists.

## Split-MNIST integration

Each task contributes 20 centroids fitted across both digits. For prediction, the classifier receives the **original normalized image**, the three selected nonshared columns plus two shared columns, their saved certificates, and an all-zero composer query. Only the router’s view of the image is deskewed under the default profile. [MNIST profiles](../hibacaml/data/mnist.py)

Set `ROUTER_MODE` at the top of the runner to `DESKEWED_EXTENDED_TANGENT` (default), `SHIFT_ROTATION_TANGENT` (fast control), or `None` (disabled). Run `python -m experiments.split_mnist`. Each boundary writes `task_<k>/router_bank.npz` alongside the matching model checkpoint directory. In `run_summary.json`, `routing.boundaries` compares routed validation predictions with the same classifier given the example’s own support. `routing.sequence` averages seen-task macro metrics over T1–T4, and `routing.final` reports T4. [Split-MNIST runner](../experiments/split_mnist.py)

## Split-MNIST validation findings

We compared five candidates on two seed-0 histories, one using a flattened image token and one using 7×7 patches (patch7). The papers supplied ideas for the candidates. The findings below come from our experiments.

| Candidate | Split-MNIST finding | Inspiration |
|---|---|---|
| R1 · Nearest pixel prototype | Led the five approaches on sequence accuracy for both histories. | [S-Prompts](https://papers.nips.cc/paper_files/paper/2022/hash/25886d7a7cf4e33fd44072a0cd81bf30-Abstract-Conference.html) |
| R2 · Highest fitted density | An image could have much higher likelihood under the wrong support’s density model. | [Deep Streaming LDA](https://arxiv.org/abs/1909.01520) and [CN-DPM](https://arxiv.org/abs/2001.00689) |
| R3 · Lowest reconstruction error | Helped on fours and struggled on zeros. Consulting R3 when R1 was unsure remained a candidate. Always fusing their scores gave no consistent sequence gain. | [Expert Gate](https://arxiv.org/abs/1611.06194) |
| R4 · Lowest binary output entropy | The classifier could be confidently wrong under a candidate support, so the lowest entropy often picked the wrong columns. | [Supermasks in Superposition](https://proceedings.neurips.cc/paper/2020/hash/ad1f8bb9b51f023cdc80cf94bb615aa9-Abstract.html) |
| R5 · R1 shortlist, R4 decision | R1’s top two usually included the image’s own support, but entropy often chose the other one. | Our combination of R1 and R4 |

Holding the same ten centroids per support fixed, we compared squared Euclidean, cosine, and bounded one-sided tangent distance. Cosine gave no consistent improvement. Allowing the centroid to shift by up to one pixel along each axis and rotate by up to 10° led at every boundary with a support choice in both histories. At T4, the number of 4/5 images sent to the 8/9 support fell from 122 to 85 on flattened inputs and from 72 to 42 on patch7.

## Why Split-MNIST defaults to deskewed extended tangents

On validation, deskewing together with extra tangents for scale, shear, and stroke thickness improved routing on both histories. The study tested these changes together. Raising the centroid count from 20 to 50 gained only 0.15–0.22 percentage points in sequence accuracy at about 2.6× the CPU routing time. That tradeoff favored **deskewed extended tangent** (historical D20). **Shift and rotation tangent** (T20) uses the original normalized pixels and the same 20-centroid budget.
