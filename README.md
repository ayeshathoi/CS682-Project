# CS682 Project: Deep Network Steganography

This repository contains the implementation of our **CS682 course project** on **Deep Network Steganography**, where we study how neural networks can be used to hide and recover other neural networks within their parameters.

---

## 📌 Project Overview

Deep Network Steganography focuses on embedding a complete neural network (secret model) inside another neural network (stego model) such that:

- The stego model behaves normally on a public (cover) task  
- The secret model can be recovered using a shared key  
- The presence of hidden information is not easily detectable  

We study two settings:
- **Intra-task setting**: Secret and cover models perform similar tasks  
- **Inter-task setting**: Secret and cover models perform different tasks  

---

## ⚙️ Methodology

We propose a unified pipeline consisting of three main stages:

### 1. Filter Insertion
Extra interference filters are inserted into selected locations of the network.

We evaluate three strategies for intra-task steganography:
- Gradient-Based Filter Insertion (GFI)
- Random Point Insertion (RPI)
- Weight-Based Filter Insertion (WBFI)

For inter-task steganography, we consider only the Gradient-Based Filter Insertion (GFI) strategy.

### 2. Side Information Hiding (SIH)
Structural information about inserted channels is encoded using a key-based scheme and embedded into a designated filter using LSB encoding.

### 3. Partial Optimization (POS)
- Only interference filters are updated during training  
- Original (secret) parameters are kept frozen  
- Statistical regularization is used to align stego model statistics with a clean reference model  

---

## 🧪 Experiments

We evaluate our method on standard vision benchmarks:

- CIFAR-10 (cover task in intra-task setting)
- Fashion-MNIST (secret task in intra-task setting)
- Oxford-IIIT Pet (cover task in inter-task setting)
- DnCNN-based image denoising (secret task in inter-task setting)

---

## 📊 Results Summary

- Near-lossless recovery in intra-task setting  
- Meaningful but degraded recovery in inter-task setting  
- Strong preservation of cover-task performance  
- Gradient-based insertion consistently performs best among all strategies  

---

## 👨‍💻 Authors

Ayesha Binte Mostofa, Meghashrita Das, Moumita Karmakar

---

## 📄 Project Materials

The project poster and detailed report are available here:

- 📄 [Poster](https://github.com/ayeshathoi/CS682-Project/blob/second_branch/CS682%20Poster.pdf)  
- 📘 [Report](https://github.com/ayeshathoi/CS682-Project/blob/second_branch/report.pdf)


---

## 📜 License

This project is intended for academic use only.
