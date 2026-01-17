# 📝 Word Recognition System (CRNN + CTC with Transfer Learning)

This project is a **deep learning–based word recognition (OCR) web application** built using **TensorFlow**, **CNN + BiLSTM (CRNN)** architecture, and **CTC decoding**, deployed with a **Flask web interface**.

The system recognizes **single-word images** and converts them into text with significantly improved accuracy and robustness, leveraging **transfer learning**, **data augmentation**, and **deeper sequence modeling**.
It is trained on the **Synth90k (100k) synthetic word dataset**.

---

## ✅ What’s New (v0.0.2)

- ✅ **Pretrained VGG16 backbone** (ImageNet weights)
- ✅ Transfer learning–based CRNN architecture
- ✅ Improved generalization with data augmentation
- ✅ Stable **fixed-size RGB input** pipeline
- ✅ Cleaner separation between **training (CTC)** and **inference**
- ✅ Higher accuracy on complex fonts and mixed casing

---

## 🚀 Features

- Image-based **single-word recognition**
- **Pretrained VGG16** + **BiLSTM (CRNN)** architecture
- **CTC (Connectionist Temporal Classification)** decoding
- Fixed-size input: `32 × 256 × 3` (RGB)
- Advanced data augmentation:
  - Random brightness & contrast
  - Small-angle rotation (KerasCV)
- Trained on **Synth90k (100k word images)** dataset
- TensorFlow `.keras` production model
- Lightweight **Flask web application**
- Simple HTML + CSS user interface
- Real-time inference on uploaded images

---

## 🧠 Model Architecture

### Pipeline Overview

```
Input Image (32 × 256 × 3)
        ↓
Pretrained VGG16 (ImageNet)
        ↓
Intermediate Feature Map (block3_pool)
        ↓
Sequence Reshaping (Width → Time steps)
        ↓
Dense Projection
        ↓
Bidirectional LSTM × 2
        ↓
Dense + Softmax
        ↓
CTC Decoding
        ↓
Predicted Word
```

### Key Details

- Image Height: `32 px`
- Image Width: `256 px`
- Channels: `3 (RGB)`
- Character Set:

    ```
    a–z, A–Z, 0–9
    ```

- Loss Function: `CTC Loss`
- Decoder: Greedy CTC decoding
- Backbone: **VGG16 (ImageNet pretrained)**

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

The dataset is downloaded automatically using the **Kaggle API**, making it suitable for **Google Colab**.

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
├── main.py                     # Flask application
├── utils.py                    # Image preprocessing & decoding
├── model/
│   └── synth90k_crnn.keras     # Trained VGG16-CRNN model
├── notebook/
│   └── training_pipeline.ipynb # Model training notebook
├── templates/
│   └── index.html              # Web UI template
├── static/
│   └── css/
│       └── styles.css          # UI styling
├── requirements.txt            # Dependencies
├── README.md                   # Project documentation
├── .gitignore
└── LICENSE                     # MIT License
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Kalana-S/Word-Recognition-System.git
   cd Word-Recognition-System

2. **Create virtual environment (optional)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux / macOS
    venv\Scripts\activate     # Windows

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

4. **Run the Flask application**:
   ```bash
   python main.py

- Then Then open your browser at: 
   ```bash
   http://127.0.0.1:5000

---

## 🖼️ How It Works (Inference)

1. Upload a **word image**
2. Image is resized to `32 × 256` and normalized
3. VGG16 extracts high-level visual features
4. BiLSTM models character sequences
5. CTC decoder converts predictions to text
6. Recognized word is displayed on the UI

---

## 📊 Sample Predictions

| Ground Truth | Prediction |
| ------------ | ---------- |
| proctoring   | proctoring |
| miffs        | miffs      |
| Plaguing     | Plaguing   |
| Jag          | Jag        |

The model performs well even with:
- Mixed casing
- Long words
- Complex fonts
- Noisy synthetic samples

---

## ⚠️ Limitations

- Designed for **single-word images only**
- Not optimized for full-line or paragraph OCR
- No language model (yet)

---

## 🧭 Versioning

| Version | Description                                        |
| ------- | -------------------------------------------------- |
| v0.0.1  | Baseline CRNN + CTC OCR system                     |
| v0.0.2  | Transfer learning, augmentation, improved accuracy |

---

## 🎥 App Demo (Screen Recording)

Full app workflow — UI → Input → Prediction<br>

https://github.com/user-attachments/assets/8688c6f2-be0b-48f7-8599-cdc5ce128c48

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
