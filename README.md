# 🍎 Automatic AI-Based Fruit Identification using CNN

## 📘 Project Overview
This project presents an **automated fruit classification system** powered by **Artificial Intelligence (AI)** and **Deep Learning**.  
It utilizes **Convolutional Neural Networks (CNN)** to identify and classify different types of fruits from images with high accuracy.

The system is integrated into a **web application** using the **Django framework**, allowing users to upload fruit images and instantly receive predictions with confidence levels.

---

## 🧠 Key Features
- 🔍 **Automatic fruit identification** using CNN.
- 📷 **Image upload interface** for user testing.
- ⚙️ **Pre-trained model (`.h5`)** for fast predictions.
- 🧩 **Django-based web app** integration for real-time classification.
- 📊 **High accuracy** using deep learning and image preprocessing techniques.

---

## 🧑‍💻 Technologies Used

| Category | Tools & Libraries |
|-----------|-------------------|
| **Programming Language** | Python 3 |
| **Deep Learning Framework** | TensorFlow, Keras |
| **Web Framework** | Django |
| **Data Handling & Analysis** | NumPy, Pandas |
| **Image Processing** | OpenCV, PIL |
| **Visualization** | Matplotlib, Seaborn |
| **Development Environment** | Jupyter Notebook, VS Code |

---

## 🗂 Project Structure<br>
Automatic-AI-Based-Fruit-identification-using-CNN/<br>
│<br>
├── Fruit_Identification_Using_Convolutional_Neural_Network/<br>
│ ├── admins/<br>
│ ├── users/<br>
│ ├── static/<br>
│ │ └── img/<br>
│ ├── templates/<br>
│ ├── manage.py<br>
│ ├── requirements.txt<br>
│ ├── fruit_cnn_model.h5 ← Trained CNN model<br>
│ ├── fruit.ipynb ← Model training notebook<br>
│ └── README.md ← (This file)<br>
│<br>
└── dataset/<br>
├── Training/<br>
└── Test/<br>






## 🧩 Model Architecture
The CNN model was designed to automatically learn visual features from fruit images.  
It includes:
- **Convolutional layers** for feature extraction  
- **Pooling layers** for dimensionality reduction  
- **Dropout** to prevent overfitting  
- **Dense layers** for classification  
- **Softmax output layer** for multi-class prediction  

> The model achieved a **validation accuracy of ~84%** using the Fruit Image Dataset.

---

## 🚀 How to Run the Project

### 🔧 Prerequisites
Make sure you have the following installed:
- Python 3.8+
- pip (Python package manager)
- Git

### ⚙️ Installation Steps

# Clone the repository
git clone <br>
https://github.com/rkprasad0001/Automatic-AI-Based-Fruit-identification-using-CNN.git

# Navigate into the project directory
cd Automatic-AI-Based-Fruit-identification-using-CNN/Fruit_Identification_Using_Convolutional_Neural_Network

# Install dependencies
pip install -r requirements.txt<br>

▶️ Run the Django Web App
python manage.py runserver


Then open your browser and go to:

http://127.0.0.1:8000/


Upload an image of a fruit to see the prediction result.<br>

📊 Dataset Description

The dataset used for training and testing contains multiple categories of fruits (e.g., Apple, Banana, Orange, Mango, etc.).
Each class has hundreds of images taken under different lighting conditions and backgrounds to improve model generalization.

Dataset Source: Adacel Technologies Limited

📈 Results
Metric	Value
Training Accuracy	90%
Validation Accuracy	84%
Loss	0.28
Classes	10+ Fruit Types
💡 Future Enhancements

📱 Convert model into a mobile application (using TensorFlow Lite).

🧠 Implement transfer learning with ResNet50V2 or MobileNetV2.

☁️ Deploy web app on AWS / Heroku / Render.

🌈 Improve dataset diversity for real-world use.

👨‍💻 Author

Ramakrishna Prasad Nalimela<br>

AI & Cybersecurity Enthusiast<br>

📧 rkprasad0001@gmail.com

🌐 rkprasad0001<br>

🪪 License

This project is released under the MIT License.
You are free to use, modify, and distribute this code for educational and research purposes with proper attribution.

⭐ Acknowledgements

Keras Documentation

TensorFlow

Fruit Image Dataset

Django Framework

🎯 “Combining AI and Vision to Make Everyday Classification Smarter.”
