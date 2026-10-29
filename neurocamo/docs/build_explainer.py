"""
Builds "NeuroCamo - Technical Walkthrough" (viva handout 1).

    python docs/build_explainer.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pdfkit as K  # noqa: E402

OUT = os.path.join(HERE, "NeuroCamo_Technical_Walkthrough.pdf")

FACTS = [
    ["Model", "NeuroCamo v4 - spiking CNN + attention bridge + dense mask head"],
    ["Parameters", "329,096 (0.33 M) - backbone alone is 93 K"],
    ["Input", "128 x 128 RGB, Poisson-encoded to 16 binary timesteps"],
    ["Datasets", "COD10K + NC4K - 8,884 train / 5,238 test images"],
    ["Supervision", "16 x 16 downsampled object-occupancy mask (256 labels/image)"],
    ["Best result", "38.6% mean IoU, 33.7% IoU@50, +0.288 confidence separation"],
    ["Hardware", "NVIDIA RTX 3050 Ti Laptop GPU, ~105 ms per frame"],
    ["Interface", "Flask dashboard on 127.0.0.1:8080 with live spike spectrogram"],
]

BLOCKS = []
A = BLOCKS.append

# ── 1. The problem ───────────────────────────────────────────────────────────
A(("h1", "1. What problem NeuroCamo solves"))
A(("p", "Camouflaged object detection (COD) is the task of finding a target whose "
       "colour, texture and outline have been deliberately matched to its background. "
       "It is harder than ordinary object detection for one specific reason: ordinary "
       "detectors lean on <b>boundary contrast</b>. A car has a sharp silhouette "
       "against a road. A camouflaged sniper does not - the whole point of the "
       "camouflage is to destroy that silhouette. So a detector must reason about "
       "subtle texture discontinuity and structural irregularity instead of edges."))
A(("p", "Our second constraint is deployment. The intended platform is a battery- or "
       "solar-powered edge device (Jetson-class, or eventually neuromorphic silicon "
       "such as Intel Loihi), mounted on a drone or a static perimeter post. That "
       "rules out a 60-million-parameter transformer. We need something that is "
       "small, and whose computation is <b>event-driven</b> so that it costs energy "
       "only when something changes."))
A(("p", "Those two constraints together are why we chose a <b>Spiking Neural Network "
       "(SNN)</b> rather than a conventional CNN, and they are the honest answer to "
       "\"why not just use YOLO\" - a question addressed in full in Section 9."))

A(("h2", "1.1 One-sentence summary for the panel"))
A(("p", "<b>NeuroCamo converts an image into a train of binary spikes, processes it "
       "through a 3-layer spiking convolutional network whose neurons integrate "
       "charge over 16 timesteps, fuses the temporal spike activity into a single "
       "spatial map using an attention bridge, and predicts a 16 x 16 occupancy grid "
       "from which the bounding box is read off directly.</b>"))
A(("break", None))
A(("h1", "2. Why a spiking neural network"))
A(("p", "A conventional artificial neuron computes a weighted sum and applies a smooth "
       "activation - it outputs a real number, every layer, every frame, "
       "unconditionally. A <b>spiking</b> neuron instead behaves like a biological "
       "one: incoming current accumulates on a membrane, leaks away slowly, and the "
       "neuron emits a single binary spike only when the membrane crosses a "
       "threshold, after which it resets."))
A(("h2", "2.1 The neuron model we use (Leaky Integrate-and-Fire)"))
A(("code", "V(t+1) = beta * V(t)  +  W . x(t)  -  V_th * S(t)\n"
           "S(t)   = 1 if V(t) > V_th else 0        (the spike)"))
A(("bul", [
    "<b>beta = 0.9</b> is the membrane decay (leak). 90% of the charge carries over "
    "to the next timestep, 10% leaks away. We set <b>learn_beta=True</b>, so "
    "back-propagation tunes the leak per channel - fast-decaying channels become "
    "transient edge detectors, slow ones become evidence accumulators.",
    "<b>V_th</b> is the firing threshold, left at the snnTorch default of 1.0.",
    "<b>S(t)</b> is binary. This is what makes the network cheap: a spike is a 1-bit "
    "event, so the expensive multiply-accumulate of a normal CNN degenerates into an "
    "<b>addition</b>, and neurons that do not spike cost nothing at all.",
]))
A(("h2", "2.2 The dead-neuron problem and the surrogate gradient"))
A(("p", "The threshold function is a step, and the derivative of a step is zero "
       "everywhere (and undefined at the step). Back-propagation through it would "
       "multiply every gradient by zero and nothing would ever learn - the "
       "<b>dead neuron problem</b>. The standard fix, and the one we use, is a "
       "<b>surrogate gradient</b>: the forward pass uses the hard step, but the "
       "backward pass substitutes the derivative of a smooth approximation. We use "
       "<b>fast_sigmoid with slope 25</b> from snnTorch."))
A(("note", "Expect to be asked this. It is the single most common SNN viva question, "
           "and the answer is: hard step forward, smooth surrogate backward."))
A(("h2", "2.3 Energy argument, stated honestly"))
A(("p", "The theoretical case is that binary sparse activations replace multiplies "
       "with adds and skip silent neurons entirely, giving one to two orders of "
       "magnitude lower energy on neuromorphic hardware. The measured caveat is "
       "that we are running on a <b>GPU</b>, which is a dense-matrix machine and "
       "gains nothing from sparsity - it must still simulate all 16 timesteps. So "
       "our 105 ms per frame is <b>slower</b> than an equivalent CNN would be on the "
       "same GPU. The energy benefit is real but only materialises on event-driven "
       "silicon. Claiming otherwise would be dishonest, and a sharp examiner will "
       "catch it."))
A(("h2", "2.4 Measured spike sparsity"))
A(("p", "The backbone reports sparsity as the fraction of neurons that stayed silent "
       "across all timesteps and layers. Our target was 85%; we currently measure "
       "roughly <b>29%</b>, i.e. about 71% of neurons fire at least once. We added a "
       "sparsity penalty to the loss (weight 0.05, target 0.85) but weighted it "
       "deliberately low, because starving the network of spikes while it was still "
       "failing to detect anything would have made the detection problem harder. "
       "Raising sparsity is listed as future work in Section 13, not claimed as "
       "achieved."))
A(("break", None))
A(("h1", "3. The pipeline, stage by stage"))
A(("p", "Five stages. Every tensor shape below is real - you can quote them."))
A(("table", ([
    ["Stage", "Module / file", "Input shape", "Output shape"],
    ["1. Poisson encode", "PoissonEncoder<br/>utils/poisson_encoder.py",
     "(1, 3, 128, 128)", "(16, 1, 3, 128, 128)"],
    ["2. Spiking CNN", "SpikingCNNBackbone<br/>models/scnn_backbone.py",
     "(16, 1, 3, 128, 128)", "spikes (16, 1, 128, 16, 16)<br/>membrane (1, 128, 16, 16)"],
    ["3. Attention bridge", "ASABridge<br/>models/attention_bridge.py",
     "spikes + membrane", "(1, 128, 16, 16)"],
    ["4. Dense head", "NeuroCamoDenseHead<br/>models/attention_bridge.py",
     "(1, 128, 16, 16)", "heatmap (1, 1, 16, 16)<br/>offset + size (1, 2, 16, 16)"],
    ["5. Decode", "decode_mask()<br/>models/attention_bridge.py",
     "heatmap logits", "box (1, 4) + confidence (1, 1)"],
], [30 * K.mm, 45 * K.mm, 38 * K.mm, 53 * K.mm])))

A(("h2", "3.1 Stage 1 - Poisson rate encoding"))
A(("p", "The image is normalised to [0, 1] and each pixel value is treated as a "
       "<b>firing probability</b>. For each of the 16 timesteps we draw a Bernoulli "
       "sample per pixel: a pixel of intensity 0.9 fires in roughly 14 of 16 "
       "timesteps, one of intensity 0.1 fires in roughly 2. The result is a binary "
       "tensor of shape (16, 1, 3, 128, 128)."))
A(("code", "x = torch.clamp(x * self.gain, 0.0, 1.0)\n"
           "x_repeated  = x.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)\n"
           "spike_train = torch.bernoulli(x_repeated)      # values in {0, 1}"))
A(("p", "This mimics the retina, which does not transmit pixel values but rates of "
       "nerve impulses. It also injects stochastic noise, which acts as a mild "
       "regulariser. We tested whether that noise was hurting us - see Section 7.4 - "
       "and it is not."))

A(("h2", "3.2 Stage 2 - the 3-layer spiking convolutional backbone"))
A(("p", "Each timestep's binary frame passes through three convolution + batch-norm + "
       "max-pool + LIF blocks. Crucially the LIF membrane state <b>persists across "
       "timesteps</b> while the convolution weights are shared, so the same filters "
       "are applied 16 times and evidence accumulates on the membranes."))
A(("table", ([
    ["Layer", "Convolution", "Resolution", "Role"],
    ["L1", "3 -> 32, 3x3", "128 -> 64", "edges, fine texture"],
    ["L2", "32 -> 64, 3x3", "64 -> 32", "camouflage boundary irregularity"],
    ["L3", "64 -> 128, 3x3", "32 -> 16", "high-level semantic structure"],
], [20 * K.mm, 40 * K.mm, 33 * K.mm, 73 * K.mm])))
A(("p", "The backbone returns three things: the layer-3 spike train over all 16 "
       "timesteps, the <b>final</b> layer-3 membrane potential, and the sparsity "
       "statistic. Only 93 K parameters live here - the whole feature extractor is "
       "smaller than a single fully-connected layer of a typical ImageNet model."))
A(("h2", "3.3 Stage 3 - ASAB, the Attention-based SNN-ANN Bridge"))
A(("p", "We now have a temporal tensor (16, 1, 128, 16, 16) but the detection head "
       "needs one spatial map. The naive fix is to average over time, which weights "
       "every timestep equally. Early timesteps, however, carry almost no "
       "information: membranes start at zero and have not yet integrated enough "
       "charge to fire meaningfully. ASAB learns that weighting instead of assuming "
       "it."))
A(("num", [
    "Average the spike train over the 16 timesteps -> (1, 128, 16, 16).",
    "Global-average-pool that to (1, 128) - one scalar firing rate per channel.",
    "Pass it through Linear(128 -> 32) -> ReLU -> Linear(32 -> 128) -> Sigmoid to get "
    "a per-channel attention weight in [0, 1].",
    "Multiply the averaged spikes by those weights, so informative channels are "
    "amplified and noisy ones suppressed.",
    "Concatenate the result with the final <b>membrane potential</b> -> (1, 256, 16, 16). "
    "Spikes are binary and lossy; the membrane is the analog sub-threshold evidence "
    "that never reached the firing threshold. A faint camouflaged edge may raise a "
    "membrane without ever triggering a spike, so discarding it loses exactly the "
    "weak signal camouflage detection depends on.",
    "Fuse with a 1x1 convolution (256 -> 128) + BatchNorm + SiLU.",
]))
A(("note", "The name means it bridges the spiking (SNN) domain and the conventional "
           "analog (ANN) domain. It comes from the base paper, Ahmed et al. 2024."))

A(("h2", "3.4 Stage 4 - the dense mask head"))
A(("p", "Two 3x3 convolutions (128 -> 96 -> 96, each with BatchNorm + SiLU) followed by "
       "three parallel 1x1 convolution heads that keep the 16 x 16 resolution:"))
A(("bul", [
    "<b>heatmap</b> (1 channel) - raw logits, per-cell \"the object covers this cell\".",
    "<b>offset</b> (2 channels) - sub-cell centre refinement, unused in mask mode.",
    "<b>size</b> (2 channels) - per-cell width/height, unused in mask mode.",
]))
A(("p", "The heatmap bias is initialised to <b>-4.595</b>, which is logit(0.01). Every "
       "cell therefore starts at probability 0.01. Without this the network begins by "
       "predicting \"object everywhere\", and since roughly 75% of cells are "
       "background the first few hundred steps are spent unlearning that instead of "
       "learning anything."))

A(("h2", "3.5 Stage 5 - decoding a box from the occupancy grid"))
A(("p", "This is the part that differs most from a standard detector, and it is worth "
       "rehearsing because it is where the design earns its keep. We do <b>not</b> "
       "regress the box. We threshold the 16 x 16 probability grid and take the "
       "bounding rectangle of the cells that survive."))
A(("code", "peak = p.max()                                  # per image\n"
           "thr  = min(0.10, peak * 0.5)                    # adaptive threshold\n"
           "m    = p >= thr\n"
           "x1, x2 = first and last column where m is True\n"
           "y1, y2 = first and last row    where m is True\n"
           "box    = [(x1 + x2 + 1)/2/W, (y1 + y2 + 1)/2/H, (x2+1-x1)/W, (y2+1-y1)/H]\n"
           "conf   = mean of the top-4 cell probabilities"))
A(("bul", [
    "The threshold is <b>adaptive</b>: min(0.10, peak x 0.5). A confident but faint "
    "detection still yields a full-extent box instead of collapsing to one cell.",
    "Confidence is the mean of the <b>top four</b> cells, not the single maximum. One "
    "hot cell is noise; a real object lights up a contiguous region.",
    "Because the box is the mask's extent, <b>width and height are never regressed</b>, "
    "so they cannot collapse to the dataset average. Section 6 explains why that "
    "matters enormously.",
]))
A(("break", None))
A(("h1", "4. Datasets"))
A(("p", "We train on two public camouflaged-object benchmarks combined. Both ship "
       "pixel-accurate segmentation masks, and both use the identical directory "
       "layout <font name='Courier'>&lt;root&gt;/&lt;split&gt;/{Image, GT_Object}</font> "
       "with .jpg images and same-stem .png masks, so no per-dataset special-casing "
       "was needed - the loader simply accepts a list of roots."))
A(("table", ([
    ["Dataset", "Train images", "of which background", "Test images", "of which background"],
    ["COD10K (Fan et al. 2020)", "6,000", "2,960 (49.3%)", "4,000", "1,974 (49.4%)"],
    ["NC4K (Lv et al. 2021)", "2,884", "0", "1,238", "0"],
    ["<b>Combined</b>", "<b>8,884</b>", "<b>2,960</b>", "<b>5,238</b>", "<b>1,974</b>"],
], [46 * K.mm, 25 * K.mm, 32 * K.mm, 23 * K.mm, 32 * K.mm])))
A(("h2", "4.1 Why the background images matter so much"))
A(("p", "Almost half of COD10K is <b>NonCAM</b> - real photographs of the same "
       "habitats containing no camouflaged animal at all, with an all-black mask. "
       "These are gold for us. A detector trained only on positives learns \"there is "
       "always something here, find it\", which produces exactly the false-positive "
       "behaviour of drawing a box on empty grass. The NonCAM frames come from the "
       "same cameras and scenes as the positives and pass through the identical "
       "preprocessing, so there is no brightness or resolution shortcut the network "
       "can exploit to separate them cheaply."))
A(("p", "The combined pool is <b>5,924 positives to 2,960 negatives</b>, a healthy 2:1 "
       "ratio requiring no resampling or class weighting."))
A(("h2", "4.2 Why we added NC4K"))
A(("p", "COD10K alone gave 3,040 positives. NC4K contributed 2,884 more - it nearly "
       "<b>doubles</b> the positive supervision, is drop-in compatible, and is "
       "distributionally different enough (different photographers, more human-made "
       "and military-style camouflage) to improve generalisation rather than just "
       "adding redundant samples."))
A(("h2", "4.3 Labels are derived, not hand-drawn"))
A(("p", "We never annotate boxes by hand. The loader reads the segmentation mask, "
       "thresholds it at 127, and derives both supervision signals from it:"))
A(("num", [
    "The <b>bounding box</b> is the extent of the white region, converted to "
    "normalised [x_centre, y_centre, width, height]. Used for evaluation only.",
    "The <b>16 x 16 occupancy mask</b> is the binary mask area-averaged down to the "
    "detector grid with cv2.INTER_AREA. Area-averaging (not nearest-neighbour) means "
    "a cell half-covered by the object gets the soft value 0.5 rather than being "
    "forced to 0 or 1. This is the actual training target.",
    "An empty mask returns width = height = 0, which makes the <b>has_object</b> flag "
    "0.0 and marks the frame as a true negative. Section 5 is the story of what "
    "happened when this line was wrong.",
]))
A(("h2", "4.4 Augmentation"))
A(("p", "Applied to the training split only, and always to image <b>and</b> mask "
       "together where geometric: horizontal flip (50%), Gaussian blur with kernel 3 "
       "or 5 (30%, simulating atmospheric haze), and brightness/contrast jitter "
       "(50%, alpha 0.8-1.2, beta -20 to +20). We deliberately do <b>not</b> use "
       "colour jitter or hue rotation: camouflage is defined by colour agreement "
       "between object and background, and shifting hue would destroy the very "
       "signal we are asking the network to detect."))
A(("break", None))
A(("h1", "5. The label-contamination bug (the strongest story you have)"))
A(("p", "Present this deliberately. Examiners reward a team that diagnosed its own "
       "failure with measurements over a team that only reports a final number."))
A(("h2", "5.1 The symptom"))
A(("p", "The earlier model (v2) drew a box in almost the same place on every image, "
       "roughly centred, and reported ~99% confidence on <b>everything</b> - including "
       "photographs with no camouflage at all. It looked like it was working until "
       "you fed it two different images and got the same answer."))
A(("h2", "5.2 The measurement that pinned it down"))
A(("p", "We wrote a diagnostic (tools/diagnose_head.py) that runs the model over "
       "hundreds of images and reports the <b>standard deviation</b> of its "
       "predictions and the <b>correlation</b> between prediction and ground truth:"))
A(("table", ([
    ["Quantity", "Measured (v2)", "Should have been"],
    ["Std-dev of predicted centre", "0.005", "~0.09 (the ground-truth spread)"],
    ["Correlation, prediction vs truth", "0.051 / -0.133 / 0.074 / 0.010", "well above 0.5"],
    ["Confidence on real targets", "0.998", "high"],
    ["Confidence on empty backgrounds", "0.996", "low"],
    ["Separation between the two", "+0.003", "large and positive"],
], [58 * K.mm, 55 * K.mm, 55 * K.mm])))
A(("p", "A near-zero standard deviation with near-zero correlation is the signature of "
       "a model that has <b>stopped looking at the input</b> and is emitting a "
       "constant. Confidence separation of 0.003 confirmed it could not distinguish a "
       "camouflaged animal from bare ground."))
A(("h2", "5.3 The root cause - one line"))
A(("p", "In the dataset loader, the function that derives a box from a mask returned a "
       "<b>fake non-zero box</b> when the mask was empty:"))
A(("code", "# BEFORE (wrong)\nif not rows.any():\n"
           "    return [0.5, 0.5, 0.1, 0.1]     # centre, 10% of the frame\n\n"
           "# AFTER (correct)\nif not rows.any():\n"
           "    return [0.5, 0.5, 0.0, 0.0]     # null box -> has_object = 0.0"))
A(("p", "Downstream, presence was inferred from the box size: "
       "<font name='Courier'>has_object = float(bbox[2] &gt; 0.01 and bbox[3] &gt; 0.01)</font>. "
       "Since 0.1 &gt; 0.01, every one of the <b>2,960 empty background images became a "
       "positive example</b> asserting \"an object is present, dead centre, occupying "
       "10% of the frame, confidence 1.0\"."))
A(("p", "That is 49.3% of every training epoch teaching a constant answer. The optimal "
       "response to contradictory labels of that kind is precisely to predict the "
       "average and ignore the image - so the network was not broken, it was "
       "correctly solving the wrong problem we had accidentally given it."))
A(("h2", "5.4 What the fix revealed"))
A(("p", "Re-evaluating v2 against corrected labels gave <b>9.6% IoU</b> against a "
       "<b>32.5%</b> \"always predict the average box\" baseline - the old model was "
       "3.4x worse than a constant. The previously reported figures had been inflated "
       "by the same contamination they were measuring. Corrected ground truth also "
       "showed real objects occupy about 45-49% of the frame, not the 26% the "
       "poisoned statistics suggested."))
A(("note", "Lesson to state out loud: we could not have found this from the loss curve. "
           "The loss was falling. Only distribution statistics of the predictions "
           "exposed it."))
A(("break", None))
A(("h1", "6. Three head designs, and why we ended on the third"))
A(("p", "Fixing the labels was necessary but not sufficient. The architecture of the "
       "output head had to change twice more."))
A(("h2", "6.1 Design A - global MLP regression (v2, abandoned)"))
A(("p", "Pool the 16 x 16 feature map to a 128-vector, then Linear(128 -> 64 -> 4) + "
       "Sigmoid to emit the box. The flaw is structural: this is a "
       "<b>mean-seeking estimator</b>. When the target centre is distributed around "
       "0.5, the loss-minimising constant output is 0.5, and if the features are not "
       "informative enough to beat that constant, gradient descent will happily sit "
       "there forever. Averaging two candidate locations produces a box between them, "
       "which is a valid compromise under L1 loss and a nonsense answer in reality."))
A(("h2", "6.2 Design B - CenterNet-style centre heatmap (v4-dense, abandoned)"))
A(("p", "Predict a 16 x 16 heatmap whose target is a Gaussian blob at the object's "
       "centre, plus per-cell offset and size maps; take the box from the argmax cell. "
       "This removes the collapse failure mode structurally, because averaging two "
       "candidate peaks gives a <b>lower</b> heatmap score, not a compromise box."))
A(("p", "It trained, but stalled. Four epochs on 8,884 images:"))
A(("table", ([
    ["Epoch", "Train loss", "Val loss", "IoU", "Mean-box baseline", "Confidence sep."],
    ["1", "3.5256", "3.6309", "25.8%", "37.0%", "+0.017"],
    ["2", "3.1542", "3.3110", "26.2%", "37.0%", "+0.014"],
    ["3", "3.1231", "3.3399", "26.3%", "37.0%", "+0.011"],
], [17 * K.mm, 24 * K.mm, 22 * K.mm, 20 * K.mm, 37 * K.mm, 32 * K.mm])))
A(("p", "IoU below the constant-box baseline, gaining 0.5% per epoch, and confidence "
       "separation <b>shrinking</b>. The size head sat frozen at 0.21 - which we "
       "verified is exactly the optimal constant for the dataset, i.e. it had "
       "collapsed to the mean again."))
A(("h2", "6.3 Why it stalled - the diagnosis, in three eliminations"))
A(("num", [
    "<b>Not the architecture or the gradient path.</b> We overfit a deliberate "
    "24-image subset: IoU reached 76.1%, heatmap loss fell 4.466 -> 0.006, and "
    "predicted sizes [0.624, 0.579] matched ground truth [0.623, 0.586]. A model that "
    "can memorise can learn; the plumbing is correct.",
    "<b>Not the Poisson encoding noise.</b> We swept the timestep count and removed "
    "the stochasticity entirely: T=16 gave 26.8%, T=32 gave 26.3%, T=64 gave 26.0%, "
    "and noise-free analog input gave 23.4%, all against a 38.5% baseline. Sampling "
    "noise is irrelevant to this failure - and notably the analog version was "
    "<b>worse</b>, so the spikes are not the handicap.",
    "<b>It is underfitting, not overfitting.</b> Train 3.1231 against val 3.3399 is a "
    "gap of about 7%. An overfitting model shows a large gap; this model could not fit "
    "even the training set.",
]))
A(("p", "Conclusion: the bottleneck was <b>supervision density</b> against a 93 K-parameter "
       "backbone. The Gaussian-centre target labels essentially <b>one cell per "
       "image</b> as positive. 8,884 images therefore provide about 8,884 positive "
       "cell labels in total - far too weak a signal to teach spatial reasoning."))
A(("h2", "6.4 Design C - mask supervision (v4-seg, current)"))
A(("p", "The insight is that both datasets ship full pixel masks, and the box-only "
       "pipeline was <b>throwing them away</b>. If the same 16 x 16 heatmap channel is "
       "trained as a downsampled <b>object occupancy mask</b> instead of a centre dot, "
       "then all <b>256 cells</b> carry a label - a 256x increase in gradient signal "
       "per image, at <b>zero extra compute</b>, with no change to the model at all."))
A(("bul", [
    "Same 329 K parameters, same 16 timesteps, same runtime.",
    "Width and height come from the mask's extent, so they are <b>never regressed</b> "
    "and cannot collapse to the dataset mean.",
    "The heatmap becomes a genuine saliency map, which is also the most convincing "
    "thing to show on screen during the demo.",
]))
A(("note", "The honest constraint behind this choice: widening the backbone to ~581 K "
           "parameters was the textbook fix for underfitting, but it needed roughly "
           "five hours of training. Mask supervision bought a similar gain in "
           "27 minutes. Engineering under a deadline is a legitimate answer."))
A(("break", None))
A(("h1", "7. How we trained it"))
A(("h2", "7.1 The loss function"))
A(("p", "Two complementary terms on the 16 x 16 occupancy grid, plus a sparsity nudge:"))
A(("code", "bce   = binary_cross_entropy_with_logits(logits, target)\n"
           "p     = sigmoid(logits).flatten(1);  t = target.flatten(1)\n"
           "inter = (p * t).sum(1)\n"
           "dice  = 1 - (2*inter + 1) / (p.sum(1) + t.sum(1) + 1)\n"
           "sp    = relu(0.85 - measured_sparsity)\n"
           "loss  = 1.0*bce + 1.0*dice.mean() + 0.05*sp"))
A(("bul", [
    "<b>BCE</b> is per-cell and stable, but it is dominated by background: objects "
    "average about a quarter of the grid and a third of frames are pure background, "
    "so a model predicting \"all background\" already scores well.",
    "<b>Soft Dice</b> is scale-invariant - it measures overlap as a ratio, so a small "
    "object contributes as much as a large one. This is what stops small targets from "
    "being ignored. The +1 in numerator and denominator is Laplace smoothing so an "
    "all-empty frame gives dice = 0 rather than dividing by zero.",
    "<b>Sparsity penalty</b> is one-sided: it only activates when sparsity falls below "
    "0.85, and is weighted at 0.05 so it cannot dominate detection.",
]))
A(("h2", "7.2 Optimisation settings"))
A(("table", ([
    ["Setting", "Value", "Reason"],
    ["Optimiser", "AdamW, weight decay 1e-4",
     "decoupled decay behaves better than L2-in-Adam"],
    ["Learning rate", "2e-3 max, OneCycleLR, pct_start 0.3",
     "warm up 30%, then anneal; suits very short runs"],
    ["Batch size", "16", "fits 4 GB VRAM with a 16-timestep unrolled graph"],
    ["Gradient clipping", "max-norm 5.0", "surrogate gradients can spike"],
    ["Epochs", "4 (then +12 resumed)", "wall-clock budget, not a principled choice"],
    ["Validation", "655-frame strided subset", "deterministic and class-mixed"],
], [30 * K.mm, 47 * K.mm, 68 * K.mm])))
A(("p", "The validation subset uses a fixed <b>stride</b> rather than the first N "
       "frames, because the Test folder sorts all CAM images before all NonCAM ones - "
       "taking the first 655 would have produced an all-positive validation set that "
       "could not measure false positives at all. Striding keeps it deterministic, so "
       "epoch-to-epoch numbers are directly comparable."))
A(("h2", "7.3 Checkpoint selection"))
A(("p", "We do <b>not</b> select on validation loss. Loss can improve while detection "
       "gets worse. We select on a composite:"))
A(("code", "score = mean_IoU + 0.5 * confidence_separation"))
A(("p", "This forces a checkpoint to be good at <b>both</b> localising the object and "
       "telling object-present from object-absent frames. When resuming training we "
       "also initialise the \"best so far\" value from the resumed checkpoint's own "
       "score, so a bad early epoch can never overwrite a good saved model."))
A(("h2", "7.4 Measured cost"))
A(("bul", [
    "0.665 s per batch of 16 -> 555 batches per epoch -> about 369 s of training per "
    "epoch, plus roughly 30 s of validation.",
    "Roughly <b>6.7 minutes per epoch</b> on an RTX 3050 Ti Laptop GPU (4 GB).",
    "Inference: about <b>105 ms</b> per frame including Poisson encoding and all 16 "
    "timesteps.",
]))
A(("note", "The 16 timesteps are the cost driver: one forward pass is effectively 16 "
           "convolutional forward passes with shared weights."))
A(("break", None))
A(("h1", "8. Results, and their honest limits"))
A(("table", ([
    ["Metric", "v2 (global MLP)", "v4-dense (Gaussian)", "v4-seg (mask, final)"],
    ["Mean IoU", "9.6%", "26.3%", "<b>38.6%</b>"],
    ["IoU@50", "~2%", "~7%", "<b>33.7%</b>"],
    ["Mean-box baseline", "32.5%", "37.0%", "37.0%"],
    ["Confidence separation", "+0.003", "+0.011", "<b>+0.288</b>"],
    ["Validation loss", "n/a (different loss)", "3.3399", "0.9389"],
    ["Parameters", "221,257", "329,096", "329,096"],
], [40 * K.mm, 32 * K.mm, 36 * K.mm, 37 * K.mm])))
A(("h2", "8.1 What these numbers mean in plain terms"))
A(("bul", [
    "<b>Mean IoU 38.6%</b> - on average the predicted box overlaps the true box by "
    "about 39% of their combined area. It finds the right region; the box is loose.",
    "<b>IoU@50 33.7%</b> - about a third of detections would count as correct under "
    "the standard PASCAL VOC criterion of 50% overlap.",
    "<b>Mean-box baseline 37.0%</b> - the score obtained by ignoring the image entirely "
    "and always predicting the average training box. Any honest detector must beat "
    "this. We beat it, but not yet by a wide margin, and we report it precisely "
    "because most student projects quietly omit it.",
    "<b>Confidence separation +0.288</b> - the gap between mean confidence on "
    "object-bearing frames and on empty frames. This rose from +0.003 to +0.288, a "
    "96-fold improvement, and it is the metric that actually fixed the visible "
    "false-positive behaviour.",
]))
A(("h2", "8.2 The decoder's measured ceiling"))
A(("p", "We validated the decoder independently of the network: we fed the <b>ground "
       "truth</b> 16 x 16 masks in as if they were predictions and measured the box "
       "that came out. This isolates decoding error from learning error."))
A(("table", ([
    ["Decode threshold", "0.05", "0.10", "0.20", "0.35", "0.50"],
    ["Mean IoU", "77.4%", "<b>77.4%</b>", "73.5%", "66.0%", "60.2%"],
    ["IoU@50", "94.8%", "<b>93.1%</b>", "86.5%", "75.4%", "64.0%"],
], [36 * K.mm, 22 * K.mm, 22 * K.mm, 22 * K.mm, 22 * K.mm, 22 * K.mm])))
A(("p", "Two conclusions. First, this swept our default threshold from 0.35 to "
       "<b>0.10</b>, an immediate free gain of 11 IoU points that cost no training. "
       "Second, <b>77.4% is the ceiling</b> of this design - even a perfect network "
       "cannot exceed it, because a 16 x 16 grid over a 128-pixel image means each cell "
       "is 8 pixels wide and the box can only ever be accurate to that quantum. Our "
       "38.6% is therefore about half of the achievable maximum, and the remaining gap "
       "is a training-budget problem, not a design dead end."))
A(("h2", "8.3 State this limitation before you are asked"))
A(("p", "The model was trained for <b>4 epochs</b>. That is not a converged result; it "
       "is what fitted the available time. The loss was still falling and IoU still "
       "rising when we stopped. We are reporting a working system with a clear "
       "improvement path, not a finished benchmark contender - and published COD "
       "methods reach far higher numbers using 30-60 million parameters and days of "
       "training on multiple GPUs."))
A(("break", None))
A(("h1", "9. How this differs from YOLO and other detectors"))
A(("p", "Expect this question first and answer it structurally, not defensively."))
A(("table", ([
    ["Aspect", "YOLO (v5/v8)", "NeuroCamo"],
    ["Neuron model", "Analog, continuous activations, one pass",
     "Spiking LIF, binary events, 16 timesteps with persistent membrane state"],
    ["Parameters", "7 M (YOLOv8s) to 68 M (YOLOv8x)",
     "0.33 M - roughly 20x to 200x smaller"],
    ["Arithmetic", "Dense multiply-accumulate everywhere",
     "Accumulate-only where a spike occurs; silent neurons cost nothing"],
    ["Target hardware", "GPU / TPU, dense tensor units",
     "Neuromorphic (Loihi, SpiNNaker) or low-power edge"],
    ["Objects per image", "Many, with anchors and non-max suppression",
     "One dominant region - single-target by design"],
    ["Classes", "80 (COCO) or user-defined",
     "One: camouflaged / not camouflaged"],
    ["Supervision", "Hand-drawn boxes", "Segmentation masks, downsampled to 16 x 16"],
    ["Output primitive", "Regressed box offsets from anchors",
     "Occupancy grid; the box is derived from its extent"],
    ["Trained on", "COCO, millions of images, many GPU-days",
     "8,884 images, 4 epochs, one laptop GPU, 27 minutes"],
], [30 * K.mm, 55 * K.mm, 60 * K.mm])))
A(("h2", "9.1 The fair comparison, stated plainly"))
A(("p", "<b>A YOLO model fine-tuned on COD10K would almost certainly beat our 38.6% "
       "IoU.</b> Say this before the panel does. It has 20-200x the parameters and the "
       "benefit of ImageNet/COCO pretraining. Our contribution is not \"more accurate "
       "than YOLO\"; it is <b>a camouflage detector that runs on event-driven "
       "hardware at 0.33 M parameters</b>, which is a different design point on the "
       "accuracy-energy curve. If accuracy at any energy cost were the goal, an SNN "
       "would be the wrong choice and we would say so."))
A(("h2", "9.2 Versus dedicated COD architectures (SINet, SegMaR, transformers)"))
A(("p", "Published COD methods (SINet, SINet-V2, SegMaR, HitNet, and recent "
       "transformer-based models) typically use a ResNet-50 or PVT backbone, "
       "ImageNet pretraining, multi-stage search-and-identify modules, and 448x448 or "
       "larger input. They report far better numbers than we do. They also use "
       "30-100 M parameters and are not deployable on a battery-powered sensor. We "
       "borrowed their <b>problem formulation</b> - treat COD as segmentation, not box "
       "regression - which is exactly the pivot described in Section 6.4."))
A(("h2", "9.3 Versus a plain CNN of the same size"))
A(("p", "This is the sharpest possible question and we have partial data on it: our "
       "own ablation replaced the Poisson spike input with noise-free analog input and "
       "scored <b>worse</b> (23.4% vs 26.8%). That is not a full ANN-vs-SNN comparison "
       "- the neurons were still LIF - and we have <b>not</b> trained an equivalent "
       "non-spiking CNN. If asked whether an ANN of 0.33 M parameters would do better, "
       "the correct answer is: probably slightly better in accuracy, worse in energy on "
       "neuromorphic hardware, and we have not measured it. That ablation is the "
       "first item of future work."))
A(("h2", "9.4 Versus SAM / foundation models"))
A(("p", "Segment Anything and similar models segment beautifully but are prompted - "
       "they answer \"what is at this point\", not \"is anything hidden here\" - and are "
       "630 M+ parameters. They solve a different problem at a wholly different "
       "deployment cost."))
A(("break", None))
A(("h1", "10. The dashboard - what every panel means"))
A(("p", "The web interface is a single-page Flask application served on "
       "<font name='Courier'>127.0.0.1:8080</font>. It is bound to localhost only, so "
       "it is not reachable from the network."))
A(("h2", "10.1 Left column - controls"))
A(("bul", [
    "<b>Drop zone</b> - upload a JPG/PNG/BMP/WEBP up to 25 MB. Large images are "
    "downscaled server-side before encoding.",
    "<b>Sample frames</b> - two built-in images, one camouflaged and one ordinary, so "
    "the contrast can be demonstrated without hunting for files.",
    "<b>Spike bands shown</b> - SPARSE / ACTIVE / SATURATED toggles that filter the "
    "spectrogram colour map by firing density. Hidden bands render grey. This is a "
    "<b>visualisation filter only</b>; it does not change the model's computation.",
    "<b>Run detection</b> - submits the frame to /api/detect.",
    "<b>Model panel</b> - which checkpoint is loaded, device, epoch, validation loss, "
    "parameter count, timestep count. Check this first if results look wrong: if it "
    "says best_model_v2.pth you are looking at the old broken model.",
]))
A(("h2", "10.2 The metric strip"))
A(("table", ([
    ["Readout", "What it actually is"],
    ["Gated confidence", "Final confidence after any scene gate. For the v4 dense head "
     "the gate is disabled, so this equals the pre-gate value."],
    ["Pre-gate confidence", "The network's raw objectness - the mean of the four "
     "highest cell probabilities in the 16 x 16 grid."],
    ["Laplacian variance", "Texture complexity of the input, computed with OpenCV. A "
     "classical image statistic, not a network output."],
    ["Silent neurons", "Measured spike sparsity - the fraction of neurons that never "
     "fired across all 16 timesteps. Currently around 29%."],
    ["Encode + 16 steps", "End-to-end latency in milliseconds, about 105 ms."],
    ["Rate > 1%", "How many of the 128 layer-3 channels are meaningfully active."],
], [42 * K.mm, 110 * K.mm])))
A(("note", "Be ready to explain that the Laplacian variance and the old scene gates are "
           "<b>classical computer vision, not the SNN</b>. They existed only because the "
           "v2 head emitted 0.998 confidence on every input, so a hand-written filter "
           "was the only thing separating a hit from a miss. Now that the dense head "
           "produces real objectness, those gates are <b>bypassed</b> - they were "
           "actively destroying correct detections in flat or bright scenes such as "
           "snow or a sky-lined target."))
A(("h2", "10.3 Input frame / Detection panels"))
A(("p", "Left is the frame as submitted, with its pixel dimensions. Right is the "
       "annotated result: the bounding box, the verdict badge, and the 16 x 16 "
       "objectness heatmap upscaled and overlaid. Three verdict bands are used - "
       "<b>DETECTED</b> above 0.60 confidence (green), <b>POSSIBLE TARGET</b> from 0.25 "
       "to 0.60 (amber), and <b>NO CAMO DETECTED</b> below 0.25 (red, box suppressed)."))
A(("p", "The heatmap is the honest showpiece. It is <b>the exact tensor the box was "
       "decoded from</b>, not a separately computed explanation, so the overlay "
       "mathematically cannot disagree with the box. That is a stronger claim than a "
       "post-hoc saliency method such as Grad-CAM can make."))
A(("h2", "10.4 Layer-3 spike spectrogram"))
A(("p", "A 128 x 16 matrix - one row per layer-3 channel, one column per timestep - "
       "where brightness is that channel's firing rate at that timestep. It is built "
       "by averaging the spike tensor over the 16 x 16 spatial grid."))
A(("p", "It is the direct analogue of an audio spectrogram's frequency-bin by "
       "time-frame matrix, and it is the panel that makes the network's temporal "
       "behaviour visible: you can point at the left edge and note that early "
       "timesteps are quiet because membranes have not yet charged, which is precisely "
       "the imbalance that ASAB learns to correct."))
A(("h2", "10.5 Live IP-camera feed"))
A(("p", "The dashboard can pull an MJPEG stream from a phone running an IP-camera app "
       "and run detection frame by frame via /api/detect_stream and /api/video_feed. "
       "This demonstrates that the pipeline is not restricted to still images."))
A(("break", None))
A(("h1", "11. Working of every Python file"))
A(("p", "Two folders. <font name='Courier'>neurocamo/</font> is the research project - "
       "model, datasets, training. <font name='Courier'>neurocamo-web/</font> is the "
       "demo, which imports the research code unchanged rather than duplicating it."))
A(("h2", "11.1 utils/poisson_encoder.py"))
A(("p", "Turns one static image into a spike train. The pixel intensity is clamped to "
       "[0, 1], repeated T=16 times, and each copy is passed through "
       "<font name='Courier'>torch.bernoulli</font>. A pixel of 0.7 therefore fires in "
       "roughly 11 of the 16 timesteps - intensity becomes firing probability, which is "
       "rate coding. <font name='Courier'>forward_sequence()</font> handles video, where "
       "each real frame supplies one timestep instead of repeating a single frame."))
A(("code", "x = torch.clamp(x * self.gain, 0.0, 1.0)<br/>"
           "x_repeated = x.unsqueeze(0).repeat(self.timesteps, 1, 1, 1, 1)<br/>"
           "spike_train = torch.bernoulli(x_repeated)"))
A(("h2", "11.2 models/scnn_backbone.py"))
A(("p", "Three convolution + LIF blocks. Each block is Conv2d -> BatchNorm -> "
       "<font name='Courier'>snn.Leaky</font>, and the whole stack is stepped 16 times "
       "inside a Python loop over the timestep axis while the membrane potentials "
       "persist between steps. Returns the layer-3 spike tensor (T, B, 128, 16, 16), the "
       "final membrane potential (B, 128, 16, 16), and the measured sparsity. beta=0.9 "
       "is learnable, and the surrogate gradient is "
       "<font name='Courier'>fast_sigmoid(slope=25)</font>. Only 93 K parameters."))
A(("h2", "11.3 models/attention_bridge.py"))
A(("p", "Three classes matter here. <b>ASABridge</b> collapses the time axis using "
       "learned channel attention and fuses the result with the membrane potential. "
       "<b>NeuroCamoDenseHead</b> is two 3x3 convolutions at width 96 followed by three "
       "1x1 heads for heatmap, offset and size, with the heatmap bias initialised to "
       "-4.595 so every cell starts at p = 0.01. <b>NeuroCamoDenseModel</b> chains "
       "backbone -> ASAB -> head and carries the two decoders, "
       "<font name='Courier'>decode</font> (peak cell + offset + size) and "
       "<font name='Courier'>decode_mask</font> (extent of the above-threshold cells). "
       "<b>NeuroCamoFullModel</b> is the old global-MLP model, kept only so the v2 "
       "checkpoints still load."))
A(("h2", "11.4 datasets/cod10k_dataset.py"))
A(("p", "Walks the COD10K and NC4K folders, pairs each image with its ground-truth PNG "
       "mask, and returns four tensors: the resized 128 x 128 image, the bounding box in "
       "normalised xywh, the objectness label, and - when "
       "<font name='Courier'>return_mask=True</font> - the mask average-pooled to "
       "16 x 16. That last tensor is the entire reason v4 works. Augmentation is "
       "horizontal flip and mild scale/crop only; no colour jitter, because hue and "
       "saturation similarity between target and background <i>is</i> the signal."))
A(("h2", "11.5 datasets/negative_dataset.py"))
A(("p", "Supplies the NonCAM negatives - real scenes with no camouflaged object - so "
       "the confidence head has something to say no to. Without these the objectness "
       "score is meaningless because every training sample would be positive."))
A(("h2", "11.6 training/train_seg.py (the run that produced our checkpoint)"))
A(("p", "Loads both datasets with masks, builds the dense model in "
       "<font name='Courier'>mask_mode=True</font>, and optimises BCE + soft Dice on the "
       "16 x 16 heatmap with a one-sided sparsity penalty. AdamW at 2e-3 with OneCycleLR, "
       "batch 16, gradient clipping at norm 5. Validation uses a strided 640-frame "
       "subset for speed, and reports IoU against the <b>mean-box baseline</b> so an "
       "improvement cannot be faked. Checkpoints are kept on the composite score "
       "<font name='Courier'>iou + 0.5 * sep</font>, which prevents a model that "
       "guesses one big box everywhere from winning on IoU alone. "
       "<font name='Courier'>NEUROCAMO_RESUME</font> continues from existing weights, "
       "and when resuming the peak learning rate is cut to 30% so the OneCycle warm-up "
       "cannot destroy the solution being resumed."))
A(("h2", "11.7 training/train_dense.py"))
A(("p", "The previous attempt: same architecture, centre-Gaussian supervision, so "
       "roughly one labelled cell per image. Kept in the repository because it is the "
       "controlled comparison that proves supervision density - not architecture - was "
       "the bottleneck. Its <font name='Courier'>iou_xywh</font> helper is imported by "
       "train_seg.py, so the two runs are scored by identical code."))
A(("h2", "11.8 neurocamo-web/inference.py"))
A(("p", "The bridge. It adds the research folder to <font name='Courier'>sys.path</font>, "
       "reads the checkpoint's <font name='Courier'>head</font> field to decide which "
       "model class to build <i>before</i> loading weights, and holds the encoder and "
       "network in a single long-lived <font name='Courier'>ModelBundle</font> so the "
       "cost is paid once at start-up. "
       "<font name='Courier'>run_inference()</font> calls the backbone explicitly rather "
       "than <font name='Courier'>model(spikes)</font>, because that keeps the layer-3 "
       "spike train available for the spectrogram. It returns the annotated image plus "
       "the JSON payload: verdict, confidence, heatmap bytes, sparsity, latency, box in "
       "both pixel and normalised coordinates."))
A(("h2", "11.9 neurocamo-web/app.py"))
A(("table", ([
    ["Route", "Purpose"],
    ["GET /", "Serves the dashboard page."],
    ["GET /api/status", "Checkpoint, device, epoch, val loss, parameter and timestep counts."],
    ["POST /api/detect", "Accepts one uploaded frame, returns the annotated PNG and metrics."],
    ["POST /api/detect_stream", "Same, for a frame pulled from a live camera stream."],
    ["GET /api/video_feed", "Proxies the MJPEG camera stream."],
], [46 * K.mm, 106 * K.mm])))
A(("p", "Binds <font name='Courier'>127.0.0.1:8080</font> with "
       "<font name='Courier'>debug=False, threaded=True</font>. Uploads are capped and a "
       "413 handler returns a clean JSON error. "
       "<font name='Courier'>sanitize_json</font> strips NaN and infinity so the browser "
       "never receives invalid JSON, and cache headers are set to no-store so a stale "
       "result is never shown."))
A(("note", "If asked about security, answer plainly: the server binds loopback only, so "
           "it is not reachable from the network, but <b>/api/detect has no "
           "authentication</b>. That is acceptable for a local demo and would need a "
           "token plus TLS before any deployment."))
A(("h2", "11.10 Front end and diagnostics"))
A(("bul", [
    "<font name='Courier'>static/index.html</font> - panel layout and markup.",
    "<font name='Courier'>static/styles.css</font> - the dark palette, shared with our "
    "sibling project so the two look like one body of work.",
    "<font name='Courier'>static/app.js</font> - uploads the frame, draws the "
    "spectrogram and heatmap onto canvases from the quantised byte arrays, and animates "
    "the metric strip. All rendering is client-side; the server sends numbers, not charts.",
    "<font name='Courier'>tools/diagnose_head.py</font> - measures whether the head has "
    "collapsed to the dataset mean: predicted-versus-ground-truth standard deviation and "
    "correlation per coordinate. This is the script that caught the v2 failure.",
    "<font name='Courier'>tools/probe_centre.py</font> - shifts a synthetic target around "
    "the frame and checks that the predicted centre follows it, i.e. that the model is "
    "translation-sensitive at all.",
]))

A(("break", None))
A(("h1", "12. How to run the demo"))
A(("num", [
    "Activate the virtual environment and start the server: "
    "<font name='Courier'>python neurocamo-web/app.py</font>. Open "
    "<font name='Courier'>http://127.0.0.1:8080</font>.",
    "<b>Check the model panel first.</b> It must read best_model_v4.pth, epoch 4, "
    "329,096 parameters. If it reads v2, an older server process is still holding port "
    "8080 - kill it and restart. This actually happened to us during testing.",
    "Run the ORDINARY SCENE sample first. Low confidence, no box. This establishes that "
    "the detector can say no, which is the claim the negatives dataset supports.",
    "Run the CAMOUFLAGED sample. Point at the heatmap and say the box was decoded from "
    "exactly that tensor.",
    "Open the spectrogram and note the quiet left edge - membranes charging - then "
    "explain that ASAB learns to reweight timesteps for precisely this reason.",
    "Upload an unseen image, ideally one the panel supplies. Do not hide a loose box; "
    "say the grid is 16 x 16 over 128 pixels, so one over-fired cell widens the box by "
    "about 6%, and that more epochs tighten it.",
]))
A(("note", "Have the numbers ready and say them without being asked: 38.6% IoU against a "
           "37.0% mean-box baseline, 33.7% IoU@50, +0.288 confidence separation, 0.33 M "
           "parameters, 4 epochs. The weakest number is the epoch count, and owning it "
           "is far better than being caught by it."))

A(("break", None))
A(("h1", "13. Limitations and future work"))
A(("h2", "13.1 What we are not claiming"))
A(("bul", [
    "<b>We are not state of the art.</b> Dedicated COD networks reach 80%+ on these "
    "datasets with 100x our parameters. Our design point is accuracy per parameter on "
    "event-driven hardware.",
    "<b>The energy argument is theoretical on our hardware.</b> A GPU gains nothing from "
    "spike sparsity, so our 105 ms is slower than an equivalent CNN. The saving is real "
    "only on neuromorphic silicon, and we have not measured it there.",
    "<b>Measured sparsity is ~29%, not the 85% we target.</b> The penalty weight of 0.05 "
    "is too gentle. This is an honest gap between goal and result.",
    "<b>Four epochs.</b> Validation loss was still falling when we stopped. The decoder "
    "ceiling test says 77.4% IoU is reachable with this exact architecture, so roughly "
    "half our remaining error is simply unspent training budget.",
    "<b>Single object.</b> decode_mask returns one box around all above-threshold cells, "
    "so two animals in one frame produce one box spanning both.",
]))
A(("h2", "13.2 What we would do next, in priority order"))
A(("num", [
    "Train to convergence - 30 to 50 epochs. The cheapest large gain available, and it "
    "needs no code change.",
    "Raise the grid from 16 x 16 to 32 x 32. Cell size drops from 8 px to 4 px, which "
    "directly attacks the quantisation error that makes boxes loose.",
    "Connected-component labelling on the mask instead of a single global extent, giving "
    "genuine multi-object output.",
    "Raise the sparsity weight and re-measure, so the energy claim rests on our own "
    "numbers rather than the literature's.",
    "Deploy to real neuromorphic hardware - Loihi or SpiNNaker - and measure joules per "
    "frame. That is the experiment that would turn our motivation into a result.",
]))
A(("gap", 10))
A(("h2", "One-paragraph summary, if you only remember one thing"))
A(("p", "NeuroCamo detects camouflaged objects with a 0.33 M-parameter spiking neural "
       "network. Images become Poisson spike trains, three spiking convolution layers "
       "extract features over 16 timesteps, an attention bridge collapses time into one "
       "spatial map, and a dense head predicts a 16 x 16 occupancy mask from whose "
       "extent the bounding box is read. Training the heatmap as a downsampled mask "
       "instead of a centre point raised IoU from 9.6% to 38.6% and confidence "
       "separation from +0.003 to +0.288, because it gave 256 labelled cells per image "
       "instead of one. Every number quoted is measured against a mean-box baseline on a "
       "held-out test split."))

K.build(OUT, K.cover(
    "NeuroCamo",
    "Camouflaged Object Detection with a Spiking Neural Network",
    "Technical walkthrough - architecture, datasets, training and results",
    FACTS,
) + K.render(BLOCKS), "Technical Walkthrough")

print(f"[OK] wrote {OUT}")











