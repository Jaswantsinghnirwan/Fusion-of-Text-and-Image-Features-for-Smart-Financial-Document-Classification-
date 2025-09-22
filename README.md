Fusion of Text and Image Features for Smart Financial Document Classification
📌 Overview

This project implements a multimodal deep learning model to classify financial documents such as balance sheets, cash flow statements, income statements, tax forms, and notes.
It combines Natural Language Processing (NLP) and Optical Character Recognition (OCR) for robust feature extraction, achieving 90.9% accuracy.

The system reduces manual document processing, minimizes errors, and has strong potential for real-time deployment in financial institutions.

⚡ Features

Automated classification of financial documents.

Text preprocessing with NLTK and spaCy (tokenization, lemmatization, stopword removal).

Embeddings with Word2Vec / TF-IDF for semantic understanding.

OCR-based text extraction for scanned documents.

Data augmentation & SMOTETomek for class imbalance handling.

Deep learning model with Bidirectional LSTM trained using TensorFlow/Keras.

Achieved 90.9% accuracy on benchmark dataset.

🛠️ Tech Stack / Tools Used

Language: Python

Deep Learning: TensorFlow, Keras

NLP: NLTK, spaCy, Word2Vec

OCR: Tesseract OCR

Data Handling: Pandas, NumPy, BeautifulSoup

Balancing: imbalanced-learn (SMOTETomek)

Visualization: Matplotlib, WordCloud

📊 Dataset

Dataset: Financial Document Classification Dataset (Kaggle)
and photos downloaded from various diffrent sources of financial documents

Contains HTML financial documents and scanned files covering multiple categories.

🚀 Model Architecture

Text Branch: Word2Vec embeddings → Dense Layers.

Image Branch (OCR extracted text): Cleaned & normalized features.

Fusion: Concatenation of text + OCR features.

Classifier: Bidirectional LSTM + Dense + Softmax.
✅ Average Accuracy: 90.9%(paticular document)

🖥️ How to Run
1. Clone the repo
git clone https://github.com/your-username/financial-document-classification.git
cd financial-document-classification

2. Install dependencies
pip install -r requirements.txt

3. Download dataset
kaggle datasets download -d gopiashokan/financial-document-classification-dataset
unzip financial-document-classification-dataset.zip

4. Train model
python financial_document_classification.py

5. Run inference
python financial_document_classification.py --predict sample.html

🔮 Future Work

Integration with transformer models (BERT, LayoutLM).

Handwriting recognition support.

Real-time deployment via FastAPI or Streamlit.

Cloud deployment for enterprise-scale usage
