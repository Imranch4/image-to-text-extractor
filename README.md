# image-to-text-extractor

```markdown
# 🖼️ image-to-text-extractor

Advanced Python GUI application to extract text from images using Tesseract OCR with preprocessing, multi-language support, and output in TXT or JSON.

---

## 🚀 Introduction

**image-to-text-extractor** is a user-friendly Python GUI tool that leverages Tesseract OCR to extract text from images. With robust preprocessing options, multi-language support, and flexible output formats, it’s designed for researchers, students, and professionals who need accurate text extraction from images of any kind.

---

## ✨ Features

- **Intuitive GUI**: Effortlessly load images and extract text.
- **Tesseract OCR Integration**: Utilizes the powerful Tesseract engine for reliable recognition.
- **Image Preprocessing**: Enhance images before extraction for better accuracy.
- **Multi-language Support**: Extract text in various languages.
- **Flexible Output**: Save results as `.txt` or `.json`.
- **Fast & Threaded Processing**: Efficient extraction even for large images.
- **Preview Functionality**: View images and results directly within the app.

---

## ⚡ Installation

### Prerequisites

- Python **3.6+**
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed and added to your system path
- Required Python packages: `pytesseract`, `opencv-python`, `Pillow`

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/image-to-text-extractor.git
   cd image-to-text-extractor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If `requirements.txt` is missing, install manually:)*  
   ```bash
   pip install pytesseract opencv-python Pillow
   ```

3. **Ensure Tesseract is installed:**
   - [Download Tesseract](https://github.com/tesseract-ocr/tesseract)
   - Add its executable to your system PATH.

---

## 🖥️ Usage

1. **Run the application:**
   ```bash
   python image-to-text.py
   ```

2. **Using the GUI:**
   - Click **“Load Image”** to select your image file.
   - Choose preprocessing options (e.g., grayscale, thresholding).
   - Select the OCR language.
   - Click **“Extract Text”** to process.
   - View the extracted text in the preview window.
   - Save results as `.txt` or `.json` using the export feature.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

Please follow the [Contributor Covenant](https://www.contributor-covenant.org/) code of conduct.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

> **Made with ❤️ using Python and Tesseract OCR**
```


## License
This project is licensed under the **MIT** License.

---
🔗 GitHub Repo: https://github.com/Imranch4/image-to-text-extractor
