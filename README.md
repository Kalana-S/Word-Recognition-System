# 📝 Word Recognition System (Hybrid CRNN OCR)

- This project is a **deep learning–based word recognition (OCR) system** that combines **multiple CRNN models** into a **hybrid inference pipeline** to improve robustness across diverse visual conditions.
- Version **v0.0.3** introduces a **confidence-aware hybrid OCR strategy**, integrating:
  - A **baseline CRNN model** (grayscale, variable-width)
  - A **transfer learning CRNN model** (VGG16-based, fixed-size RGB)
- The system dynamically selects the most reliable prediction at inference time, resulting in **higher real-world accuracy without retraining**.
- The application is deployed using a **Flask web interface** and trained on the Synth90k **synthetic word dataset**.

---

## ✅ What’s New (v0.0.3)

- ✅ **Hybrid OCR pipeline** (multi-model inference)
- ✅ Confidence-based model selection
- ✅ Improved robustness to:
  - Stylized fonts
  - Color backgrounds
  - Mixed casing
  - Slight rotations
- ✅ Refined **CTC confidence estimation**
- ✅ No changes required to UI or Flask logic
- ✅ Backward-compatible with previous models

---

## 🚀 Features

- Image-based **single-word recognition**
- Hybrid inference using two **CRNN models**
- **Confidence-aware decision logic**
- CTC-based sequence decoding
- Supports:
  - Grayscale & RGB inputs
  - Fixed-width and variable-width pipelines
- TensorFlow `.keras` production models
- Flask-based web interface
- Lightweight and modular codebase

---

## 🧠 Model Architecture

### Hybrid Inference Overview

```
Input Word Image
        ↓
Preprocessing
  ├── Grayscale (Baseline CRNN)
  └── RGB Fixed Size (Transfer Learning CRNN)
        ↓
CRNN Models (parallel)
        ↓
CTC Decoding + Confidence Scoring
        ↓
Best Prediction Selection
        ↓
Final Recognized Word
```

---

## 🧩 Model Details

### Baseline CRNN Model (v0.0.1)

- Input: Variable width, grayscale
- CNN + BiLSTM (CRNN)
- CTC decoding
- Strong on:
  - Simple fonts
  - Clean backgrounds
  - Short words

### Transfer Learning CRNN Model (v0.0.2)

- Input: `32 × 256 × 3` (RGB)
- Backbone: VGG16 (ImageNet pretrained)
- BiLSTM × 2
- Strong on:
  - Stylized fonts
  - Color backgrounds
  - Rotated or complex images

---

## 🧪 Hybrid Decision Strategy

At inference time:

1. The **baseline CRNN** predicts first
2. A **CTC confidence score** is computed
3. If confidence ≥ threshold → accept result
4. Otherwise → fallback to **VGG16-CRNN**

This approach:

- Avoids overfitting to one model
- Preserves speed for easy cases
- Improves accuracy for difficult samples

---

## 🔠 Character Set

- Character Set:
  ```
  a–z, A–Z
  ```

- Case-sensitive recognition
- No language model or dictionary constraints

---

## 🏋️ Dataset

- **Dataset Name:** Synth90k (Synthetic Word Dataset)
- **Images:** 100,000 word images
- **Labels:** Stored in `labels.txt`
- **Format:**

    ```
    00000.jpg slinking
    00001.jpg REMODELERS
    00002.jpg Chronographs
    ```

- The dataset is downloaded using the **Kaggle API**, making it suitable for **Google Colab**.

---

## 🧰 Technologies Used

- **Python**
- **TensorFlow / Keras**
- **VGG16 (Transfer Learning)**
- **BiLSTM (CRNN)**
- **CTC Decoding**
- **KerasCV** – Data augmentation
- **Flask** – Web server
- **HTML / CSS** – Frontend UI
- **NumPy**
- **Kaggle API** – Dataset download
- **Google Colab** – Model training

---

## 📁 Project Structure

```
├── main.py                                             # Flask application
├── utils.py                                            # Preprocessing & CTC decoding
├── model/
│   ├── baseline_crnn.keras                             # Baseline CRNN model
│   └── transfer_learning_crnn.keras                    # VGG16-based CRNN model
├── notebook/
│   ├── training_pipeline_basline.ipynb                 # Baseline Colab Pipeline
│   └── training_pipeline_transfer_learning.ipynb       # Transfer Learning Colab Pipeline
├── templates/
│   └── index.html                                      # Web UI template
├── static/
│   └── uploads/                                        # Uploaded images
├── requirements.txt                                    # Dependencies
├── README.md                                           # Project documentation
├── .gitignore
└── LICENSE                                             # MIT License
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kalana-S/Word-Recognition-System.git
   cd Word-Recognition-System

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

3. **Run the Flask application**:
   ```bash
   python main.py

4. **Access the Web UI:** 
   ```bash
   http://127.0.0.1:5000

---

## 🖼️ How It Works (Inference)

1. Upload a **single-word image**
2. Image is preprocessed for both models
3. Each model predicts independently
4. CTC decoding generates text
5. Confidence-aware selection chooses best result
6. Final word is displayed with model info

---

## 🧭 Versioning

| Version | Description                                |
| ------- | ------------------------------------------ |
| v0.0.1  | Baseline CRNN + CTC OCR                    |
| v0.0.2  | VGG16 transfer learning CRNN               |
| v0.0.3  | Hybrid OCR with confidence-based selection |

---

## 🎥 App Demo (Screen Recording)

Full app workflow — UI → Input → Prediction<br>

https://github.com/user-attachments/assets/42aa983f-8604-4a94-8d4e-e7af5c239b1c

---

## 🤝 Contribution

Contributions are welcome.

- Fork the repository
- Create a feature branch
- Submit a pull request

---

## 📜 License

This project is licensed under the **MIT License** <br>
See the `LICENSE` file for details.
