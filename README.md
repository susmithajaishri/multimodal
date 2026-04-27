# Multimodal Fusion for Multi-Level Political Meme Classification in Tamil

This repository contains the implementation of a multimodal machine learning framework designed for the DravidianLangTech@ACL 2026 shared task. The project focuses on classifying the stance and targets of political memes in Tamil and Malayalam.

## 🚀 Overview
The system integrates visual and textual semantics to decode complex satirical intent in social media content. It employs a hierarchical classification approach:
- **Level 1 (Stance):** Binary classification (Support/Praise vs. Troll/Oppose).
- **Level 2 (Target):** Multi-class identification (Individual, Party, or Group).

## 🏗️ Architecture
The model utilizes a parallel encoding strategy followed by feature fusion:
- **Visual Encoder:** CLIP (ViT-B/32) extracting 512-dimensional image embeddings.
- **Textual Encoder:** XLM-RoBERTa base capturing 768-dimensional linguistic context.
- **Fusion Layer:** A Multi-Layer Perceptron (MLP) with Batch Normalization, ReLU activation, and Dropout (0.4).

## 📊 Dataset statistics
To address data scarcity, we implemented a cross-lingual strategy:
- **Training Set:** 1,228 memes (728 Tamil + 500 Malayalam).
- **Test Set:** 201 Tamil memes.

## 🛠️ Requirements
- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Pytesseract (OCR)
- PIL (Pillow)
- Scikit-learn, Pandas, Numpy

## 💻 Usage
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/susmithajaishri/multimodal.git](https://github.com/susmithajaishri/multimodal.git)
