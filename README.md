# 🩺 An EfficientNet-Based Framework for Automated Skin Lesion Diagnosis  
## With Clinical and Patient-Centric Report Generation

---

## 📌 Project Overview

Skin cancer is one of the most common and dangerous diseases worldwide, where early and accurate diagnosis is crucial for successful treatment. Manual analysis of dermoscopic images is challenging due to visual similarity between lesion types, noise, hair artifacts, illumination variations, and dependence on expert knowledge.

This project presents an **end-to-end AI-based system** for **automated skin lesion diagnosis** using **deep learning**, **lesion segmentation**, **explainable AI (Grad-CAM)**, and **GPT-based report generation**.  
The system is designed to assist both **doctors and patients** by providing accurate predictions along with transparent and understandable explanations.

---

## 🎯 Objectives

- Automate detection and classification of skin lesions from dermoscopic images  
- Improve diagnostic accuracy using lesion segmentation  
- Provide explainable predictions using Grad-CAM heatmaps  
- Generate clinical reports for doctors and simplified reports for patients  
- Build a real-world, deployable AI-based healthcare solution  

---

## 🧠 System Architecture

The proposed framework consists of the following components:

### 1. Image Preprocessing
- Noise removal  
- Hair artifact removal  
- Illumination and contrast normalization  

### 2. Lesion Segmentation
- Separates lesion region (ROI) from background skin  
- Helps the classifier focus on the affected area  
- Improves overall classification accuracy  
<img width="1408" height="768" alt="u-net architecture" src="https://github.com/user-attachments/assets/a41760f7-87a8-499c-aa58-321a7fa8e87d" />

### 3. Skin Lesion Classification
- EfficientNet-based Convolutional Neural Network  
- Extracts discriminative features such as color, texture, and shape  
- Classifies lesions into multiple disease categories with confidence scores

### 4. Explainable AI (Grad-CAM)
- Generates heatmaps highlighting important image regions  
- Improves model transparency and trust  
- Helps doctors understand why a prediction was made  

### 5. GPT-Based Report Generation
- **Doctor Report**: diagnostic reasoning, confidence score, and risk level  
- **Patient Report**: simple, non-technical explanation of the result  

### 6. Final Output
- Predicted disease class  
- Confidence score  
- Segmentation mask  
- Grad-CAM heatmap  
- Downloadable medical and patient-friendly reports  

---

## 🧬 Skin Lesion Classes

The system supports classification of the following skin lesion categories:

- Melanoma (MEL)  
- Melanocytic Nevus (NV)  
- Basal Cell Carcinoma (BCC)  
- Benign Keratosis (BKL)  
- Actinic Keratosis / Intraepithelial Carcinoma (AKIEC)  
- Dermatofibroma (DF)  
- Vascular Lesions (VASC)  

---

## 🛠️ Technologies Used

### Machine Learning & Deep Learning
- Python  
- TensorFlow / Keras  
- EfficientNet  
- Convolutional Neural Networks (CNN)  
- Grad-CAM (Explainable AI)  

### Backend & Application
- Flask (or Django)  
- OpenCV  
- NumPy  
- Matplotlib  

### NLP & Report Generation
- GPT / Large Language Models  
- Prompt-based medical report generation  
- LangChain (optional)  

### Datasets
- HAM10000  
- ISIC  
- PH2  

---

## 📊 Performance Highlights

- High classification accuracy on HAM10000 dataset  
- Improved performance with segmented images  
- Strong generalization on PH2 dataset  
- Clinically interpretable predictions using Grad-CAM  

---

## 🧪 How to Run the Project

### Note : Need to install Ollama and a local model
### 1. Clone the project
### 2. Run the below command in terminal

```
streamlit run app.py
```

---

<img width="1763" height="603" alt="segmentation_and_prediction_2" src="https://github.com/user-attachments/assets/c9aa7661-b941-433f-9096-928697ae0afe" />
<img width="1909" height="910" alt="input_2" src="https://github.com/user-attachments/assets/c068b633-9c09-44ff-abe5-1c65c839ecfc" />
<img width="1781" height="658" alt="gradcam_2" src="https://github.com/user-attachments/assets/7b4f4f8d-199c-4b73-8af2-983a641876b1" />
<img width="1754" height="757" alt="doctorreport_2" src="https://github.com/user-attachments/assets/1d184943-1bdb-43f7-a9a7-9d8ab6ed7bf3" />
<img width="1770" height="837" alt="patientreport_2" src="https://github.com/user-attachments/assets/849ba653-8e5c-490e-9467-339b8a8d1641" />

---


## 🚀 Future Enhancements

- Mobile application support  
- Real-time camera-based diagnosis  
- Multi-language patient reports  
- Integration with hospital EMR systems  
- Federated learning for privacy-preserving training  

