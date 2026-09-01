# DistortAware: Robust AI-Generated Image Detection After Reposting

## Inspiration

AI-generated images are becoming increasingly photorealistic, but detecting them in their original form is only half the problem.

Images uploaded to social platforms rarely remain pristine. They are compressed, resized, cropped, filtered, sharpened, blurred, screenshotted, and reposted. These transformations can erase generator-specific artifacts or introduce new artifacts that resemble AI generation. A detector that performs well on clean images may therefore become unreliable in the environment where it is actually needed.

This led us to a more practical question:

> Can we detect AI-generated and AI-edited images while remaining reliable after realistic post-processing and redistribution?

We approached this as both a detection and a calibration problem. The system needs to find evidence of AI generation, but it also needs to understand how image processing has changed the reliability of that evidence.

This is directly relevant to social platforms, fact-checkers, moderators, journalists, and ordinary users because redistributed content is often the content most in need of verification—and the least likely to retain its original quality or provenance.

---

## What it does

DistortAware analyses an image and produces:

- a three-class prediction: **authentic**, **fully synthetic**, or **AI-tampered**;
- a binary AIGC score combining the synthetic and tampered classes;
- localized patch-level evidence showing where the model found suspicious signals;
- predicted post-processing types, such as JPEG compression, noise, blur, resizing, colour adjustment, or cropping;
- estimated distortion severity;
- a distortion-conditioned final decision threshold.

The key idea is that distortion is **not** treated as evidence that an image is fake. Authentic photographs are also compressed and edited.

Instead, distortion is treated as context:

> Given how this image has been processed, how should we interpret its AIGC score?

This distinction allows the model to compensate for score shifts caused by redistribution without automatically penalizing compressed or low-quality photographs.

Architecturally, the primary detector finds AIGC evidence, the distortion estimator identifies how the image has been transformed, and a bounded threshold adapter adjusts how the detector’s score is interpreted. Physics and residual-artifact modules are then attached as independent explanations without being allowed to alter the primary result.

---

## How we built it

Our final system grew out of three stages of experimentation.

### 1. Difference-in-Difference reconstruction evidence

Our first approach was based on the paper [Difference-in-Difference for AI-generated image detection](https://arxiv.org/pdf/2602.23732).

A pretrained diffusion model reconstructs an input image \(x_0\). We first use DDIM inversion and sampling to obtain a reconstruction \(x_0'\), following [Denoising Diffusion Implicit Models](https://arxiv.org/pdf/2010.02502).

The first-order reconstruction error is:

$$
\Delta = |x_0 - x_0'|
$$

Earlier reconstruction-based detectors such as [DIRE](https://arxiv.org/pdf/2303.09295) classified images directly from this residual. However, a single reconstruction error contains both meaningful distance-to-manifold information and reconstruction noise.

DID reconstructs the image a second time. We treat $x_0'$ as a new input, invert and reconstruct it, and obtain $x_0''$:

$$
\Delta' = |x_0' - x_0''|
$$

The second-order Difference-in-Difference feature is:

$$
\Delta^2 = \Delta - \Delta'
$$

Each reconstruction introduces perturbation from inversion, sampling, and model imperfections. This perturbation tends to be similar across the two reconstruction passes.

For an AI-generated image already close to the diffusion manifold:

$$
\Delta \approx |\delta(x_0)|,\qquad
\Delta' \approx |\delta(x_0')|
$$

Because $x_0$ and $x_0'$ are already close, these terms tend to cancel:

$$
\Delta^2 \approx 0
$$

For a real photograph, there is a larger gap between the input and what the diffusion model naturally represents. That gap survives the subtraction, producing a stronger second-order residual.

We trained separate classifiers over $\Delta$ and $\Delta^2$, using their agreement as the final prediction.

We tested:

- Stable Diffusion 1.5 and SANA-1.6B as reconstructors;
- ResNet-18 and ResNet-50 as residual classifiers.

SANA was slightly more robust than Stable Diffusion 1.5, suggesting that a stronger reconstruction manifold can produce more stable forensic evidence. Surprisingly, ResNet-18 performed better than ResNet-50. The larger classifier appeared more likely to overfit residual patterns that did not generalize.

DID gave us an important insight: increasing model size does not automatically improve forensic generalisation. The representation of the evidence matters more than classifier capacity.

However, diffusion reconstruction was computationally expensive, and its cross-dataset performance remained sensitive to the source of the real and generated images.

### 2. Spatial evidence with PatchHead

Our second approach was inspired by [PatchHead](https://arxiv.org/pdf/2608.09223).

Instead of repeatedly reconstructing the image, we pass it through a frozen DINOv3 vision transformer. DINOv3 represents the image as a grid of patch tokens, preserving localized visual information.

We then reshape those tokens into a spatial feature map and classify them using a lightweight CNN-based head. This produces both:

- a global image prediction; and
- a patch-level evidence map.

We adapt DINOv3 using rank-8 LoRA modules rather than fine-tuning the entire backbone. This keeps the number of trainable parameters small while allowing the representation to specialize for image forensics.

PatchHead was substantially faster than diffusion reconstruction and performed strongly on both clean and transformed images. It also gave us localized evidence that could be visualized and inspected.

However, our robustness experiments revealed a more subtle failure mode.

Under severe JPEG compression and Gaussian noise, PatchHead often retained strong AUC while its classification accuracy decreased. For example, our baseline produced:

| Condition | Accuracy | AUC | Real accuracy | Fake accuracy |
|---|---:|---:|---:|---:|
| Clean | 94.5% | 0.984 | 89.8% | 96.5% |
| JPEG quality 30 | 83.0% | 0.965 | 93.2% | 78.7% |
| Noise $\sigma=0.05$ | 83.5% | 0.950 | 88.1% | 81.6% |
| Noise $\sigma=0.10$ | 86.5% | 0.933 | 78.0% | 90.1% |

This was an important clue.

The high AUC meant the model could still rank real and generated images reasonably well. The lower accuracy meant that the score distribution had shifted relative to the fixed decision threshold.

The detector had not completely lost its evidence—it had become miscalibrated.

### 3. Distortion-aware detection

This observation led to DistortAware.

We extended PatchHead so it learns two related tasks:

1. detect whether an image is authentic, synthetic, or AI-tampered;
2. estimate how the image has been transformed.

During training, our augmentation pipeline applies realistic post-processing and returns exact metadata describing the transformation type and magnitude.

The supported transformations are:

- JPEG compression;
- Gaussian blur;
- resizing;
- Gaussian noise;
- brightness, contrast, and saturation adjustment;
- cropping;
- compositions of multiple transformations.

We deliberately included stronger transformations than our original pipeline, including JPEG quality down to 25 and Gaussian noise up to $\sigma=0.11$. This prevents the model from seeing only mild corruption during training and then being expected to extrapolate to severe benchmark conditions.

We also discovered and fixed a bug in our original Gaussian-noise augmentation. It added one random scalar to the entire image instead of sampling independent noise for every pixel and channel. Correcting this significantly improved the realism of our training distribution.

### External robustness results

We evaluated baseline PatchHead and DistortAware on the same held-out external
benchmark: 500 real COCO images and 500 DALL-E generated images, evaluated
under seven post-processing conditions. Each condition contains 1,000 images,
and every completed run returned all expected predictions with no missing
records or inference errors.

| Condition | Baseline accuracy | DistortAware accuracy | Change | Baseline ROC-AUC | DistortAware ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Clean | 88.2% | 94.9% | +6.7 pp | 0.955 | 0.990 |
| JPEG quality 90 | 90.3% | 95.3% | +5.0 pp | 0.965 | 0.992 |
| Gaussian blur | 82.4% | 88.5% | +6.1 pp | 0.923 | 0.963 |
| Gaussian noise | 85.9% | 89.8% | +3.9 pp | 0.942 | 0.977 |
| Colour jitter | 88.2% | 95.5% | +7.3 pp | 0.955 | 0.993 |
| Crop | 82.1% | 89.1% | +7.0 pp | 0.913 | 0.961 |
| Resize to 0.5× | 80.5% | 75.8% | -4.7 pp | 0.907 | 0.881 |
| **Unweighted mean** | **85.4%** | **89.8%** | **+4.5 pp** | **0.937** | **0.965** |

For clean images, the other standard classification metrics also improve:

| Model | Precision | Recall | F1 |
|---|---:|---:|---:|
| Baseline PatchHead | 85.8% | 91.6% | 88.6% |
| DistortAware | 95.5% | 94.2% | 94.9% |

DistortAware improved performance in six of seven conditions. The largest gains
were under colour jitter, cropping, blur, and clean inputs. On clean images,
ROC-AUC improved from 0.955 to 0.990, while precision, recall, and F1 all
improved.

Resize was the important exception. DistortAware produced more false positives
on resized real COCO images, reducing real-image accuracy from 69.2% to 58.0%.
Therefore, the method is more robust overall, but it is not uniformly robust:
resize needs targeted augmentation or calibration before deployment claims.

---

## Blind distortion estimation

At deployment time, we do not know which transformation was applied. The model must estimate it directly from the image.

Our distortion estimator combines two complementary sources of information.

### Learned visual features

DINOv3 features provide semantic and visual context. They help the model distinguish true corruption from naturally complex textures such as grass, fabric, hair, foliage, or low-light sensor noise.

### Analytical forensic features

We also compute low-level measurements including:

- high-pass residual energy;
- horizontal and vertical gradient statistics;
- Laplacian energy;
- local sharpness and noise measurements;
- approximate block-boundary discontinuities associated with JPEG compression.

The estimator predicts multiple distortion types because real redistribution pipelines often compose operations. For example, an image may be resized, JPEG-compressed, and colour-adjusted.

It also predicts normalized severity values for each operation.

The analytical features directly expose low-level artifacts, while the learned features provide the context needed to interpret them. This hybrid design is more robust than relying exclusively on handcrafted thresholds or a neural network.

---

## Distortion-conditioned threshold adjustment

The original PatchHead three-class prediction remains intact. The probabilities assigned to synthetic and tampered content are combined into a binary AIGC score.

A small threshold adapter then consumes the predicted distortion vector and produces a bounded adjustment in logit space:

$$
z_{\text{adjusted}} = z_{\text{base}} - s(d)
$$

where:

- $z_{\text{base}}$ is the original AIGC logit;
- $d$ is the predicted distortion vector;
- $s(d)$ is the learned threshold shift.

A positive shift requires stronger evidence before declaring the image AIGC. A negative shift lowers the required evidence.

The adapter is initialized to output zero, so enabling it initially reproduces the original detector exactly. It must learn useful corrections rather than starting with random changes to the decision boundary.

We also detach the distortion vector before passing it to the threshold adapter. This prevents the authenticity loss from redefining “distortion” into an arbitrary hidden feature. The distortion estimator remains grounded by explicit type and severity supervision.

During training, known augmentation metadata is used as scheduled guidance. The model gradually transitions toward conditioning on its own predictions.

During normal evaluation and inference, it uses **predicted distortion only**.

Ground-truth transformation metadata is available only through a clearly labelled `oracle` mode. We use this as an upper-bound ablation, never as our reported deployment result.

The innovation is therefore not simply that we trained with more augmentation. We separated three questions that are often conflated:

1. What visual evidence suggests that the image is AIGC?
2. What transformations have affected that evidence?
3. How should those transformations alter the decision boundary?

This makes the model’s response to post-processing explicit, supervised, and testable.

---

## Physics-aware explanation

To complement PatchHead’s pixel-level evidence, we built an automated physics-consistency sidecar. It proposes and evaluates applicable scene constraints, including perspective and vanishing points, cast shadows, planar reflections, and structural geometry.

Each cue can report a possible inconsistency, a consistent result, or abstain when the image does not contain enough reliable evidence. This is important because many images simply do not contain visible shadows, reflections, or strong perspective lines.

The physics engine never changes the primary detector’s score or verdict. Instead, it provides a second, human-readable explanation of possible scene contradictions alongside PatchHead’s localized forensic evidence.

---

## Browser extension and end-to-end demonstration

We packaged the system into **PrismGuard — AIGC Signal Inspector**, a user-initiated Chromium browser extension. It can scan eligible images in the current viewport or perform a bounded whole-page scan that scrolls through lazy-loaded content and then restores the user’s position.

The extension captures rendered image pixels and sends them to an authenticated inference service running locally on the user’s device. It places a visible result badge over each analysed image, while the popup keeps the different evidence sources clearly separated:

- the primary detector’s verdict and classifier signal;
- a summary of PatchHead’s localized patch-evidence grid;
- applicability-aware physics results for perspective, cast shadows, and reflections, including proposal counts and reasons for abstaining;
- a three-class residual-artifact diagnostic and bounded weak evidence-mask summary.

Physics results are reported as **consistent**, **inconsistent**, **indeterminate**, or **not applicable**, rather than forcing a conclusion when the required scene evidence is absent. The residual sidecar examines RGB and high-pass signals for suspicious low-level patterns, but its localization is treated as weak supporting evidence rather than a forensic segmentation.

Both the physics and residual-artifact outputs are explicitly labelled **explanation only**. They cannot vote on, adjust, or reverse the selected primary detector’s score, threshold, or verdict. This allows users to inspect several kinds of supporting evidence without presenting any individual artifact or physical anomaly as proof of AI generation.

The extension uses only `activeTab`, `scripting`, and local `storage` permissions. It does not send page URLs, captions, cookies, or browsing history to the detector, and no paid or hosted inference API is required.

A verified launcher checks the identity of the model artifacts before starting the offline loopback services. For a reproducible detector replay, this launcher deliberately disables the optional physics and artifact profiles. The same explainability features are available when the normal local inference service is started with the relevant sidecars enabled.

For demonstration, we prepared a small WildFake evaluation wall containing COCO val2017 photographs and DALL·E Advanced images. Ground-truth labels are visible to the human viewer but remain outside the image crop received by the extension. These samples are used only for demonstration and evaluation, never for training or threshold tuning.

This turns the project from an offline model experiment into a concrete user workflow. A moderator, fact-checker, journalist, or ordinary user can inspect rendered content without manually downloading each image or sending private browsing data to an external service.

---

## Engineering for reliability and reproducibility

We separated the system into independently testable components:

- the primary PatchHead detector;
- the distortion estimator and threshold adapter;
- explanation-only physics and residual sidecars;
- an authenticated local inference API;
- the browser-extension interface.

This separation prevents an optional explainer failure from silently changing the detector’s output. The browser service validates uploads, model identity, response schemas, selected detector profiles, and authentication. It also places limits on image size, decoded pixels, cached results, page images, and scrolling depth.

Model artifacts remain outside the browser extension. The verified launcher checks their SHA-256 identities, loads the DINOv3 backbone offline, waits for authenticated health checks, and shuts down its child processes together. Invalid hashes and incomplete model configurations fail closed instead of falling back to an unverified detector.

The current integrated repository passes:

- 86 physics-engine tests;
- 18 PatchHead and unified-inference tests;
- 16 browser-service tests;
- 17 browser-extension tests;
- 5 verified-launcher tests.

These tests cover component contracts, authentication, candidate selection, bounded whole-page scanning, model-profile isolation, explanation noninterference, malformed responses, checkpoint verification, and process supervision. They demonstrate implementation reliability, but are kept separate from model-accuracy claims.

---

## Practicality beyond the prototype

Several design choices were made specifically to keep the approach buildable:

- the large DINOv3 backbone is frozen;
- only lightweight LoRA modules and task-specific heads are trained;
- PatchHead avoids repeated diffusion reconstruction during normal inference;
- the local service can run without paid inference APIs;
- browser scanning is user-initiated and bounded;
- optional explainers can be disabled when lower latency is required;
- checkpoints, manifests, evaluation splits, and model identities are explicitly recorded.

The local architecture also keeps sensitive page context out of the inference request. Only rendered image pixels are analysed; source URLs, cookies, captions, and browsing history are not required.

The prototype is not yet a production moderation system, but its detector, calibration logic, local API, explainers, and browser interface already form a working deployment path rather than a purely speculative design.

---

## Challenges we faced

### Avoiding dataset shortcuts

Our models performed strongly within individual datasets but initially collapsed toward chance when transferred between WildFake and SID_Set.

The detector was learning source-specific information: different real-image collections, camera pipelines, resolutions, compression histories, and generator families.

We addressed this by:

- canonicalizing input resolution;
- splitting data using stable image groups;
- separating training, validation, calibration, and benchmark manifests;
- training on pooled WildFake and SID data;
- evaluating cross-dataset transfer explicitly;
- reporting negative transfer results instead of hiding them.

This taught us that robustness to image transformation is not the same as generalisation to an unseen data source.

### Distinguishing noise from natural texture

High-frequency content is ambiguous. Grass, hair, fabric, dark photographs, and detailed architecture can resemble sharpening, noise, or generative texture.

Simple thresholds on residual entropy or Laplacian energy produced too many false positives. Combining analytical signals with DINOv3 features allowed the estimator to use semantic context rather than treating every textured region as suspicious.

### Maintaining calibration under severe corruption

JPEG compression and noise shifted real and generated-image scores differently. A single global threshold could therefore favor real-image recall under one condition and fake-image recall under another.

The strong AUC under these conditions showed that the ranking signal remained useful. This motivated a small calibration layer rather than replacing the entire detector.

### Preventing training leakage

During augmentation, we know exactly which transformation was applied. During deployment, we do not.

It would have been easy to report results using known transformation labels, but that would not represent a usable detector. We therefore separated predicted and oracle evaluation modes and made predicted distortion the default in every deployment and SLURM evaluation script.

### Working within hackathon compute limits

Our experiments included diffusion reconstruction, multiple classifier sizes, DINOv3, LoRA, transformation sweeps, and cross-dataset evaluation.

To remain practical, we froze the main DINOv3 backbone, trained only lightweight LoRA and task-specific heads, stored only trainable checkpoint tensors, and automated training and evaluation through reproducible SLURM workflows.

The submitted primary model remains below the two-billion-parameter limit.

---

## What we learned

The biggest lesson was that **accuracy, ranking, robustness, and generalisation are different properties**.

A detector can have high AUC but poor accuracy because its threshold is miscalibrated. It can be robust to JPEG compression within one dataset but fail on an unseen generator. It can perform well on average while producing an unacceptable false-positive rate on a particular class of real photographs.

We also learned that:

- larger classifiers can overfit forensic residuals;
- reconstruction quality can matter more than classifier size;
- analytical and neural features are stronger together;
- realistic augmentation must include correct transformation implementations and severe examples;
- pooled datasets improve generalisation, but do not prove universal detection;
- negative results are valuable because they reveal which shortcuts the model learned;
- explainability should describe evidence, not pretend to provide proof.

Most importantly, we stopped treating post-processing as an inconvenience to remove and started treating it as a variable the detector should understand.

---

## What we are proud of

We built an end-to-end prototype that includes:

- two independently implemented AIGC detection approaches;
- first- and second-order diffusion reconstruction evidence;
- a spatial DINOv3 PatchHead detector;
- lightweight LoRA adaptation;
- authentic, synthetic, and AI-tampered classification;
- distortion-labelled training augmentation;
- blind multi-label distortion estimation;
- per-image threshold calibration;
- localized patch evidence;
- automatic physics-consistency and residual-artifact explanation sidecars;
- a privacy-conscious Chromium extension and authenticated local inference service;
- a hash-verified offline demonstration workflow;
- clean, transformed, cross-dataset, and false-positive evaluation;
- reproducible manifests, unit tests, inference tools, and GPU runbooks.

Rather than selecting only our strongest result, we documented where each approach succeeded and failed. These failures directly shaped the final design.

---

## Responsible use and limitations

DistortAware produces probabilistic forensic evidence, not proof of authorship or malicious intent.

Unusual camera pipelines, heavy artistic editing, illustration, strong compression, and generators excluded from training can still cause errors. Provenance metadata may also be missing or stripped during redistribution.

Physics and residual-artifact findings are supporting explanations, not independent proof. Some images do not contain testable shadows, reflections, or perspective geometry, while low-level artifacts can also be introduced by ordinary camera processing and reposting.

For these reasons, the system should support human review and content-provenance workflows rather than automatically accuse creators. In a real deployment, thresholds should be selected according to an explicit false-positive target and monitored across image sources, regions, devices, and generator families.

---

## What’s next

Our next steps are to:

- evaluate against generator families completely excluded from training;
- add composed social-media pipelines such as resize followed by JPEG recompression;
- include screenshots and camera recapture;
- estimate uncertainty in the distortion prediction;
- suppress threshold adjustment when distortion confidence is low;
- calibrate against deployment-specific false-positive targets;
- integrate C2PA Content Credentials when available;
- retain pixel-based detection when metadata has been removed;
- distill or quantize PatchHead for lower-latency inference.

DistortAware is not intended to be the final answer to synthetic-media detection. It is a practical step toward detectors that understand not only what an image looks like, but also what has happened to it before it reaches them.

Our final results and analysis are here:

https://github.com/henrlly/DistortAware/blob/main/RESULTS.md
