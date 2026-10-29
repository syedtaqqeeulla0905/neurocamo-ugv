# Review 2: Detailed Presentation Script

**Goal:** Provide a deep technical explanation for each slide to show total mastery of the subject.

---

### **Slide 1: Title Slide**
"Good morning, respected members of the panel. We are Team [X]. Today, we present Review 2 of our major project: **'Ultra Low Power Spiking Neural Networks for Real-Time Military Camouflage Detection on Edge Devices.'** This review covers our finalized architecture and 2026-standard implementation strategy."

### **Slide 2: Contents**
"Our discussion is structured into four main phases: the Research Foundation (Literature Survey), the Architectural Design (NeuroCamo), the Technical Methodology (Training and Encoding), and finally, our Hardware Integration and Performance Targets."

### **Slide 3: Introduction**
"In modern defense, identifying camouflaged targets is a life-critical task. Traditional sensors often fail because camouflage is designed to deceive the human eye and standard AI by blending textures. Our project introduces a **neuromorphic solution** that looks beyond static pixels."

### **Slide 4: Why Military Camouflage Needs Edge AI**
"Standard cloud-based AI is unusable in combat due to signal jamming and latency. However, bringing high-power AI to the edge leads to battery failure. We need a system that can operate for 6+ hours on a single charge without losing detection accuracy."

### **Slide 5: Challenges in Deploying Detection**
"Current Deep Learning models suffer from **Thermal Throttling**. On a Jetson Nano, a standard YOLOv9 model will cause the GPU to overheat within minutes, dropping the frame rate. We solve this by moving from continuous floating-point math to discrete spiking events."

### **Slide 6-20: Literature Survey (Papers 1-20)**
"We have reviewed 20 foundational works. We analyzed how **SCOUT (2025)** uses text-guided data selection and how **Spikformer (2023)** proves that SNNs can handle global context. Most importantly, we've identified a 'Research Gap': no current work focuses on **Ultra-Low Power SNNs** specifically for **Camouflage** in 2026-standard edge hardware."

### **Slide 21: Problem Statement**
"The problem is the **Computational-Energy Tradeoff**. You shouldn't have to choose between a 'smart' robot with 30-minute battery life and a 'dumb' robot with 10-hour battery life. We need a system that is both intelligent and neuromorphically efficient."

### **Slide 22: Objectives**
"Our primary objective is to implement a **Spiking-CNN** using **Adaptive LIF neurons**. We aim to deploy this on the Jetson Nano and achieve a **mAP of 88%** on the COD10K benchmark while reducing power consumption by 70%."

### **Slide 23: Existing Hybrid SNN-ANN Architecture**
"This is our **Base Paper Architecture**. It utilizes an SNN backbone for feature extraction. The key innovation is the **Attention-based Bridge**, which converts temporal spikes into spatial feature maps that a standard detection head can understand. This gives us SNN efficiency with ANN precision."

### **Slide 24: Challenges in Current Systems (ANN)**
"Traditional ANNs suffer from the **Static Frame Bottleneck**. They process the 'empty' background in every single frame. This is a massive waste of energy. SNNs, however, are **Event-Driven**—they only consume power when a pixel 'spikes' due to texture or motion."

### **Slide 25: Challenges in Current Systems (Lightweight)**
"Even 'Lightweight' ANNs like MobileNet are not truly efficient because they use **Dense Matrix Multiplications**. SNNs replace these with **Accumulate-only (AC)** operations, which are mathematically simpler and require less hardware energy."

### **Slide 26: Proposed System (Neuromorphic Detection)**
"Our system, **NeuroCamo**, uses **Adaptive Leaky Integrate-and-Fire (ALIF)** neurons. Unlike standard LIF, ALIF neurons can learn their own 'membrane decay' rates, allowing them to adapt to different camouflage textures in real-time."

### **Slide 27: Proposed System (OPSEC)**
"From a military perspective, **OPSEC** is vital. By performing all inference **locally on the Jetson**, we eliminate all radio-frequency emissions associated with cloud data transfer. This makes the robot 'radio silent' and undetectable to enemy signal trackers."

### **Slide 28: SNN-Driven Edge Detection Pipeline (Phase 1)**
"In Phase 1, we focus on the **Single-Node Detection Pipeline**. We take the raw RGB feed, convert it to spikes, and pass it through our SNN backbone. This provides the core intelligence before we add Swarm capabilities in Phase 2."

### **Slide 29: Proposed Methodology (Flowchart)**
"This flowchart traces our engineering path. We move from the **CSI-input** to **Poisson Encoding**, into the **snnTorch** training environment, and finally to the **GPIO-actuation** layer for real-time robot response."

### **Slide 30: Data Pipeline (Temporal Encoding)**
"We utilize the **COD10K and CAMO** datasets. Each 128x128 image is converted into a **Temporal Spike Train over 16 timesteps**. By using **Poisson Rate Coding**, we ensure that high-contrast edges generate more spikes, allowing the SNN to 'see' the hidden camouflage boundary."

### **Slide 31: Model Architecture (Sparsity)**
"Our brain is a 3-layer **SCNN**. To train this, we use **Backpropagation Through Time (BPTT)** with an **ArcTan Surrogate Gradient**. This 'smooths' the spike function so we can calculate gradients. We are targeting **90% Sparsity**, meaning the hardware remains mostly silent until a threat is seen."

### **Slide 32: Robot Integration (Hardware-in-the-loop)**
"We've optimized the deployment using **NVIDIA TensorRT**. By using **FP16 Quantization**, we halve the memory bandwidth required. The **GStreamer pipeline** ensures that video frames are moved from the camera to the GPU with zero CPU involvement."

### **Slide 33: Performance Benchmarking**
"We are benchmarking against **YOLO26-N**, the 2026 industry baseline. We will use the **tegrastats** utility to perform a **Watt-per-Frame** analysis. Our goal is to prove that SNNs are significantly more 'Metabolically Efficient' than standard ANNs."

### **Slide 34: Projected Efficiency Gains (The Dashboard)**
"This tactical dashboard shows our target. Notice the **Power Draw: 1.2 Watts**. A standard AI would require 8.7 Watts to do the same task. This 7x gain is the reason why SNN is the future of edge-based defense."

### **Slide 35: Target Performance Metrics**
"Our success criteria: **>88% mAP accuracy**, **Inference Latency < 25ms**, and a **Battery life of 6+ hours**. These metrics make our system field-ready for long-duration reconnaissance missions."

### **Slide 36: Current Development Phase**
"We are in the **Active Specification Phase**. We have indexed our datasets, finalized the mathematical framework for our Poisson Encoders, and configured the **snnTorch/CUDA 12** environment on our host system."

### **Slide 37: Hardware Integration (Exploded View)**
"This exploded view shows our finalized assembly. The **Jetson Nano** is mounted on a 4WD chassis. The **L298N driver** is controlled via PWM signals from the Jetson's GPIO, allowing the robot to steer toward or away from detected threats."

### **Slide 38: Hardware Components**
"Our BOM (Bill of Materials) is optimized for 2026. We use **JetPack 6.0**, which provides the latest **TensorRT 10.x** kernels required to run our Spiking Neural Network at maximum efficiency."

### **Slide 39-45: References**
"Our project is anchored in modern research, with 20 citations from the most prestigious AI conferences, including the **2024 Ahmed et al.** paper which serves as our architectural anchor."

### **Slide 46: Thank You & Q&A**
"Thank you for your time. We have prepared a rigorous implementation plan to prove that neuromorphic AI is the best solution for military camouflage detection. We are now open for technical questions."
