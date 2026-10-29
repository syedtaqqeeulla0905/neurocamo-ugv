# Review 2: 100 Important Questions & Answers

## I. Project Overview & Motivation
1. **Q:** What is the primary objective of your project?
   **A:** To develop an ultra-low-power camouflage detection system using Spiking Neural Networks (SNNs) for real-time edge deployment.
2. **Q:** Why use SNN instead of traditional ANN like YOLO?
   **A:** SNNs process information via sparse spikes, which is significantly more energy-efficient for battery-constrained edge devices.
3. **Q:** What is the real-world application of this project?
   **A:** Autonomous stealth reconnaissance and threat detection in military field environments.
4. **Q:** What does "Ultra Low Power" mean in numbers?
   **A:** We target an inference power draw of under 1.5W, compared to 8-10W for standard ANNs.
5. **Q:** What is "Neuromorphic Computing"?
   **A:** A computing paradigm that mimics the biological brain's structure and function (neurons and spikes).
6. **Q:** Why is camouflage detection harder than normal object detection?
   **A:** High texture similarity between the foreground and background makes traditional color-based segmentation fail.
7. **Q:** What is the "NeuroCamo" solution?
   **A:** It’s our proposed framework that uses SNNs to focus on temporal texture changes to find hidden objects.
8. **Q:** How does this project support national security?
   **A:** It enables small, undetectable drones and robots to perform long-endurance surveillance.
9. **Q:** What is "Edge AI"?
   **A:** AI processing that happens locally on the device (like Jetson Nano) instead of the cloud.
10. **Q:** Why is cloud-dependency a problem for military AI?
    **A:** Signal latency, risk of interception, and lack of connectivity in remote hostile zones.

## II. Spiking Neural Networks (SNN) Basics
11. **Q:** What is a "Spike" in SNN?
    **A:** A discrete binary event (0 or 1) that represents information at a specific point in time.
12. **Q:** What are the three dimensions of SNN data?
    **A:** Width, Height, and Time (Timesteps).
13. **Q:** What is "Neural Sparsity"?
    **A:** The percentage of neurons that remain inactive (silent) at a given time.
14. **Q:** How does sparsity lead to power savings?
    **A:** Hardware consumes power only when a spike is generated; 90% sparsity means 90% power saving in theory.
15. **Q:** What is "Asynchronous Processing"?
    **A:** SNNs process information as spikes arrive, rather than waiting for a global clock cycle.
16. **Q:** Define "Membrane Potential".
    **A:** An internal variable in a neuron that accumulates input spikes until it hits a threshold.
17. **Q:** What happens when the threshold is reached?
    **A:** The neuron "fires" a spike and resets its membrane potential.
18. **Q:** What is the "Reset Mechanism"?
    **A:** After firing, the membrane potential is either set back to zero (Reset to Zero) or reduced by the threshold value (Reset by Subtraction).
19. **Q:** What is "Temporal Coding"?
    **A:** Representing information based on the exact timing of spikes.
20. **Q:** How do SNNs handle noise better than ANNs?
    **A:** The leaky nature of the neurons naturally filters out low-frequency noise.

## III. Leaky Integrate-and-Fire (LIF) Model
21. **Q:** What is the LIF neuron model?
    **A:** The standard mathematical model for spiking neurons; it integrates inputs but "leaks" potential over time.
22. **Q:** What is the "Leaky" part?
    **A:** A decay constant (beta) that reduces the membrane potential if no new spikes arrive.
23. **Q:** What is the "Integrate" part?
    **A:** Summing up the incoming weighted spikes from previous layers.
24. **Q:** What is the "Fire" part?
    **A:** Generating an output spike when $V_{mem} > V_{th}$.
25. **Q:** What is the mathematical formula for LIF?
    **A:** $U(t+1) = \beta U(t) + WX(t+1) - S(t)V_{th}$.
26. **Q:** What is "Adaptive LIF" (ALIF)?
    **A:** A model where the threshold or decay rate changes over time to adapt to different signals.
27. **Q:** Why use ALIF for camouflage?
    **A:** It allows the network to stay sensitive to subtle background variations.
28. **Q:** Define "Refractory Period".
    **A:** A brief period after firing during which a neuron cannot fire again.
29. **Q:** What is "Synaptic Weight" in SNN?
    **A:** A value that determines the strength of the influence one neuron has on another.
30. **Q:** How is LIF different from the ReLU activation in ANNs?
    **A:** ReLU is instantaneous and continuous; LIF is temporal and discrete.

## IV. Information Encoding
31. **Q:** What is "Rate Coding"?
    **A:** Converting pixel intensity into the probability of a spike (higher intensity = more spikes).
32. **Q:** What is "Poisson Spike Encoding"?
    **A:** A type of rate coding where spikes are generated following a Poisson distribution.
33. **Q:** What is "Latency Coding"?
    **A:** Representing intensity by the timing of the first spike (bright pixels fire earlier).
34. **Q:** What is "Delta Encoding"?
    **A:** Generating spikes only when a pixel value *changes* by a certain amount.
35. **Q:** Why is Delta Encoding useful for robots?
    **A:** It naturally handles motion and drastically reduces redundant background data.
36. **Q:** Which encoding are you using for the first layer?
    **A:** Poisson Rate Coding (standard for snnTorch).
37. **Q:** How many timesteps (T) are you using?
    **A:** We target $T=16$ to balance accuracy and latency.
38. **Q:** What happens if T is too small?
    **A:** The network might not have enough temporal information to detect textures accurately.
39. **Q:** What happens if T is too large?
    **A:** Inference latency increases, and power consumption goes up.
40. **Q:** How do you feed a 2D image into an SNN?
    **A:** We repeat the image across all T timesteps, adding stochastic spike noise in each step.

## V. Training & Optimization
41. **Q:** What is the "Dead Neuron Problem"?
    **A:** Since spikes are 0 or 1, the derivative is zero almost everywhere, preventing standard gradient descent.
42. **Q:** How do you solve the Dead Neuron Problem?
    **A:** By using **Surrogate Gradient Descent**.
43. **Q:** Define "Surrogate Gradient".
    **A:** A differentiable function (like ArcTan or Sigmoid) that approximates the spike function during backpropagation.
44. **Q:** What is "BPTT"?
    **A:** Backpropagation Through Time; it unfolds the SNN over all timesteps to calculate gradients.
45. **Q:** What is "STBP"?
    **A:** Spatio-Temporal Backpropagation; a specific method to train SNNs using surrogate gradients.
46. **Q:** Which surrogate function are you using?
    **A:** The **ArcTan (ATan)** function.
47. **Q:** What is the "Loss Function" for your model?
    **A:** A combination of Spike-based Cross-Entropy and Bounding Box Regression loss.
48. **Q:** How do you handle "Vanishing Gradients" in deep SNNs?
    **A:** By using learnable membrane constants and specialized weight initialization.
49. **Q:** What optimizer are you using?
    **A:** **Adam Optimizer** with a learning rate scheduler.
50. **Q:** What is "ANN-to-SNN Conversion"?
    **A:** Training a normal ANN first and then converting it to SNN by matching firing rates.

## VI. Architecture (Spiking-CNN)
51. **Q:** Describe your SNN architecture.
    **A:** A 3-layer Spiking Convolutional Neural Network (SCNN) backbone with a detection head.
52. **Q:** What is the role of the Convolutional layer in SNN?
    **A:** To extract spatial features like edges and textures using spiking filters.
53. **Q:** What is "Spike-Pooling"?
    **A:** A layer that reduces spatial resolution by picking the most active spike in a window.
54. **Q:** What is the "Detection Head"?
    **A:** The final layers that predict the bounding box coordinates (x, y, w, h).
55. **Q:** Is your detection head SNN or ANN?
    **A:** It is a **Hybrid Head**; we integrate spikes over time to create a dense feature map for regression.
56. **Q:** What is the "ASAB" module?
    **A:** Attention-based SNN-ANN Bridge; it fuses temporal spikes into meaningful spatial features.
57. **Q:** Why use 3 layers instead of 50?
    **A:** To keep inference speed high on the Jetson Nano and avoid memory overflow.
58. **Q:** What is the input resolution?
    **A:** 128x128 pixels (optimized for edge).
59. **Q:** How do you handle multi-scale detection?
    **A:** Through feature map pyramids or multi-resolution encoding.
60. **Q:** What is "Learnable Beta"?
    **A:** It means the decay rate of the LIF neuron is a parameter the network learns during training.

## VII. Datasets (COD10K & CAMO)
61. **Q:** What is the **COD10K** dataset?
    **A:** A large-scale Camouflaged Object Detection dataset with 10,000 images across 78 classes.
62. **Q:** What is the **CAMO** dataset?
    **A:** A dataset focused specifically on diverse backgrounds like jungle, desert, and snow.
63. **Q:** How many images are you using for training?
    **A:** Approximately 7,000 images for training and 2,000 for validation.
64. **Q:** What "classes" are relevant to your project?
    **A:** Humans (soldiers), vehicles, and military equipment.
65. **Q:** How do you handle "Imbalanced Data"?
    **A:** By using weighted loss functions or data oversampling.
66. **Q:** What "Augmentations" are you applying?
    **A:** Gaussian blur, random flipping, and contrast jitter.
67. **Q:** Why use Gaussian Blur?
    **A:** To simulate the atmospheric haze or fog found in field surveillance.
68. **Q:** Do you have "Negative Samples" (Background only)?
    **A:** Yes, to ensure the SNN doesn't fire spikes when no threat is present.
69. **Q:** How are the images labeled?
    **A:** Via binary masks and COCO-style bounding box annotations.
70. **Q:** Are you using any "Pre-trained" weights?
    **A:** No, we are training the SNN backbone from scratch using **snnTorch**.

## VIII. Hardware (NVIDIA Jetson Nano)
71. **Q:** Why Jetson Nano instead of Raspberry Pi?
    **A:** The Jetson has a **128-core Maxwell GPU** which is essential for parallel spike processing.
72. **Q:** What is **JetPack 6.0**?
    **A:** The latest NVIDIA SDK that provides CUDA 12 and optimized AI libraries.
73. **Q:** What is **TensorRT**?
    **A:** An SDK for high-performance deep learning inference; it optimizes the model graph for speed.
74. **Q:** What is **FP16 Quantization**?
    **A:** Reducing weight precision from 32-bit to 16-bit to double the speed on Jetson hardware.
75. **Q:** What is a **CSI Camera**?
    **A:** Camera Serial Interface; it provides a high-bandwidth, low-latency connection to the processor.
76. **Q:** Why use **GStreamer**?
    **A:** To build an optimized video pipeline that moves frames directly to GPU memory (Zero-copy).
77. **Q:** How does the Jetson talk to the motors?
    **A:** Via **GPIO (General Purpose Input/Output)** pins or I2C communication.
78. **Q:** What is the role of the **L298N**?
    **A:** It is a Dual H-Bridge motor driver that controls the speed and direction of the robot wheels.
79. **Q:** What is the voltage of your battery?
    **A:** **11.1V (3S Li-Po)**, which is standard for high-current robot motors.
80. **Q:** How do you monitor power consumption?
    **A:** Using the **tegrastats** utility on the Jetson Nano.

## IX. Performance & Benchmarking
81. **Q:** What is **mAP@0.5**?
    **A:** Mean Average Precision at an IoU threshold of 0.5; the standard accuracy metric for detection.
82. **Q:** What is **IoU**?
    **A:** Intersection over Union; it measures how much your predicted box overlaps with the real one.
83. **Q:** What is **FPS**?
    **A:** Frames Per Second; it measures the real-time speed of the system.
84. **Q:** What is your target FPS?
    **A:** **25-30 FPS** for smooth real-time surveillance.
85. **Q:** How do you calculate **Energy per Inference**?
    **A:** (Total Power Draw / FPS) = Joules per frame.
86. **Q:** What is the "Baseline" model you are comparing against?
    **A:** **YOLOv9-Tiny** or **YOLO26-N**.
87. **Q:** If SNN is so good, why isn't it used everywhere?
    **A:** High training complexity and lack of standardized "Neuromorphic hardware" in the past.
88. **Q:** How do you validate your results?
    **A:** By testing on a held-out validation set that the model has never seen before.
89. **Q:** What is a **Confusion Matrix**?
    **A:** A table used to visualize the performance of a model (True Positives, False Positives, etc.).
90. **Q:** How do you handle "Occlusions" (objects behind trees)?
    **A:** The temporal memory of SNN neurons helps keep track of targets even if they are briefly blocked.

## X. Future Scope & Security (OPSEC)
91. **Q:** What is **OPSEC** in your project?
    **A:** Operational Security; ensuring the robot processes data locally so it cannot be "hacked" remotely.
92. **Q:** Can you add **Thermal Cameras**?
    **A:** Yes, Phase 2 involves fusing RGB and Thermal spikes for night-vision camouflage detection.
93. **Q:** What about a **Swarm** of robots?
    **A:** Future scope involves multiple robots sharing threat locations via LoRa.
94. **Q:** How does **LoRa** work?
    **A:** Long-Range radio communication that uses very little power to send signals up to 10km.
95. **Q:** What is the **GPS Module** for?
    **A:** To mark the exact coordinates of a detected threat on a remote map.
96. **Q:** How will you improve accuracy in the future?
    **A:** By using **Spikformers** (Spiking Transformers) and larger datasets.
97. **Q:** What is the "Metabolic Cost" of your AI?
    **A:** It's a term for the energy spent per calculation; we aim to reduce this by 70%.
98. **Q:** Is the system waterproof?
    **A:** The current prototype is not, but future field versions will have an IP67-rated enclosure.
99. **Q:** Who are the stakeholders?
    **A:** Defense agencies, border security units, and autonomous vehicle researchers.
100. **Q:** Why should we give you a high grade for Review 2?
    **A:** Because we have moved from a proposal to a **rigorous engineering design**, selected a **2024-standard base paper**, and defined **measurable 2026 targets**.
