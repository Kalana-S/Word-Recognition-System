# 📝 Word Recognition System (CRNN + CTC)

This project is a **deep learning–based word recognition (OCR) web application** built using **TensorFlow**, **CNN + BiLSTM (CRNN)** architecture, and **CTC decoding**, deployed with a **Flask web interface**.

The system recognizes **single-word images** and converts them into text with high accuracy. It is trained on the **Synth90k (100k) synthetic word dataset**, making it robust to variations in font, casing, and word length.

---

## 🚀 Features

- Image-based **single-word recognition**
- **CNN + Bidirectional LSTM (CRNN)** architecture
- **CTC (Connectionist Temporal Classification)** decoding
- Variable-width image support (no fixed padding)
- Trained on **Synth90k (100k word images)** dataset
- TensorFlow `.keras` production model
- Lightweight **Flask web application**
- Simple HTML + CSS user interface
- Real-time inference on uploaded images

---

## 🧠 Model Architecture

### Pipeline Overview

```
Input Image (H=32, variable width)
        ↓
Convolutional Feature Extractor (CNN)
        ↓
Sequence Conversion (Width → Time steps)
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
- Width: Variable (kept proportional)
- Character Set:

    ```
    a–z, A–Z, 0–9
    ```

- Loss Function: `CTC Loss`
- Decoder: Greedy CTC decoding

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
- **CNN + BiLSTM (CRNN)**
- **CTC Decoding**
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
│   └── synth90k_crnn.keras     # Trained TensorFlow model
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
2. Image is resized proportionally to height = 32
3. CNN extracts visual features
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
- Different fonts

---

## ⚠️ Limitations

- Designed for **single-word images only**
- Not optimized for full-line or paragraph OCR
- No language model (yet)

---

## 🎥 App Demo (Screen Recording)

Full app workflow — UI → Input → Prediction<br>

https://github.com/user-attachments/assets/6a9a129f-d722-42f9-b3dc-b6a8268287a8

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
