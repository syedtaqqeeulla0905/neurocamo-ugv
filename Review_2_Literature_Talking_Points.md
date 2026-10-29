# Review 2: Literature Survey Talking Points (Slides 6-25)

Use these 1-sentence "Impact Points" to explain each paper quickly. 

---

### **Paper 1: SCOUT (Yan et al. 2025)**
"This paper is crucial because it introduces **Semi-supervised learning** using text descriptions, which helps the model understand 'what hidden objects look like' when labeled data is scarce."

### **Paper 2: Realistic Camouflage Benchmarks (Xin et al. 2025)**
"This work provides the first **Realistic Benchmark** for camouflage, moving away from simple images to complex, real-world object detection labels."

### **Paper 3: Infrared UAV Tracking (Fang et al. 2025)**
"We analyzed this paper to understand how **Dynamic Feature Refinement** helps in tracking small targets from a drone's perspective, which is applicable to our robot's height."

### **Paper 4: Camouflage Survey (Xiao et al. 2024)**
"This is the most comprehensive **2024 Survey**, which we used to map out the entire history of camouflage detection and identify the current lack of SNN-based solutions."

### **Paper 5: YOLOv9 Military (Huong 2024)**
"This paper gives us our **ANN Baseline** for military camouflage; it shows that while YOLOv9 is accurate, it is too computationally heavy for battery-powered edge robots."

### **Paper 6: Hidden Patterns (Kamble 2024)**
"A key study on using **LSTM and CNN pipelines** to extract temporal patterns, which directly informed our choice to use the temporal memory of SNNs."

### **Paper 7: Cross-Level Refinement (Ge 2024)**
"This work proves that fusing **low-level texture** with **high-level semantics** is the only way to successfully separate a camouflaged foreground from its background."

### **Paper 8: CLAD Contrastive Learning (Ge 2024)**
"This paper introduces **Contrastive Learning**, which teaches the model to focus on the 'differences' between textures rather than the colors."

### **Paper 9: CamoFocus (Khan 2024)**
"CamoFocus uses **Focal Modulation** to sharpen the edges of camouflaged objects; we aim to achieve a similar sharpening effect using Spike-based Attention."

### **Paper 10: Hybrid SNN-ANN (Ahmed et al. 2024) [BASE PAPER]**
"This is our **Foundational Base Paper**. It introduces the **Attention Bridge**, which is the core architecture we are implementing to combine SNN efficiency with ANN accuracy."

### **Paper 11: SpikeYOLO (Luo et al. 2024)**
"SpikeYOLO proves that we can run **YOLO-style detection** using pure spikes, achieving a massive reduction in power while maintaining real-time frame rates."

### **Paper 12: FCENet (Liu et al. 2024)**
"This paper demonstrates a **Lightweight Network** with only 4 million parameters, proving that we can fit high-accuracy models into the Jetson Nano’s 4GB RAM."

---

## **Recent SNN & Efficiency Works (2023-2026)**

### **Paper 13: Spikformer (Zhou et al. 2023)**
"Spikformer introduces **Spiking Self-Attention**, proving that SNNs can now handle 'global context' just like Transformers but with 1/10th the energy."

### **Paper 14: Temporal Efficient Training (Deng et al. 2023)**
"This paper provides our training strategy (**TET**), allowing us to train the model with very few timesteps ($T=8$ to $16$) for ultra-low latency."

### **Paper 15: Spiking Self-Attention (Kundu 2024)**
"Kundu's work optimizes the **Linear Spiking Attention** mechanism, ensuring that our attention bridge remains efficient even at higher camera resolutions."

### **Paper 16: Neuromorphic Tracking (Li 2024)**
"A study on using **Spike-based Temporal Difference** filters to detect moving objects, which is our primary method for finding 'motion-based' camouflage."

### **Paper 17: UAV SNN Efficiency (Chen 2024)**
"This paper shows how to use **Channel-Sparsity** to shut down unused hardware neurons, which is how we target our 1.2W power consumption goal."

### **Paper 18: Adaptive Thresholding (Wang 2025)**
"This provides the math for our **Adaptive LIF neurons**; it allows the threshold to 'float' based on the background brightness, preventing false spikes."

### **Paper 19: Event-based Fusion (Zhu 2024)**
"Zhu's work is the first to use **SNNs specifically for the COD10K dataset**, proving that spikes are mathematically superior at finding camouflage edges."

### **Paper 20: YOLO26 (Ultralytics 2026)**
"This provides our **2026 Competitive Baseline**. It allows us to compare our NeuroCamo system against the fastest non-spiking model currently in existence."

---

### **Paper 21: snnTorch Foundation (Eshraghian 2023)**
"Finally, this paper is the bible for our software framework. It explains how to use **Surrogate Gradients** to train our SNN using standard deep learning lessons."
