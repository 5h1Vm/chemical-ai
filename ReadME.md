# Chemical-AI: Forensic Chemical Color Prediction System

🏆 **Winner of ₹2 Lakh Seed Money at NFSU Forensics Hackathon**

---

## Overview

Chemical-AI is an AI-powered solution for forensic chemistry, designed to predict the color outcomes of chemical spot tests based on input reagents and analytes. This project leverages machine learning to assist in rapid, accurate, and reproducible chemical analysis—significantly enhancing the efficiency of forensic investigations.

---

## Table of Contents

- [Features](#features)
- [Background](#background)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Web Application](#web-application)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Features

- **AI-driven color prediction** for forensic spot tests
- **User-friendly web interface** for quick input and result visualization
- **Supports multiple chemical reagents and analytes**
- **Extensible dataset** for continuous improvement and retraining
- **Open source and customizable** for research and educational use

---

## Background

Forensic chemistry is essential for the scientific analysis of evidence in criminal investigations. Traditional spot tests rely on expert interpretation of color changes, which can be subjective and error-prone. Chemical-AI addresses these challenges by using AI to automate and standardize color prediction, making forensic analysis more reliable and accessible.

---

## How It Works

1. **Input**: The user selects or enters the names of the chemical reagent and analyte.
2. **Processing**: The AI model processes this input, referencing a trained dataset of known reactions and their observed colors.
3. **Output**: The predicted color outcome is displayed to the user, along with confidence scores where applicable.

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. **Clone the repository:**
   ```
   git clone https://github.com/5h1Vm/chemical-ai.git
   cd chemical-ai
   ```

2. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```
   python app.py
   ```
   The web interface will be available at `http://localhost:5000`.

---

## Usage

- Open your browser and navigate to `http://localhost:5000`.
- Enter/select the reagent and analyte.
- Click "Predict" to view the expected color result.

---

## Dataset

The model is trained on a curated dataset of forensic spot tests, including:
- Chemical names (reagent, analyte)
- Observed color outcomes

You can extend the dataset by adding new entries to `data/dataset.csv` and retraining the model.

---

## Model Architecture

- **Input Encoding**: Chemical names are tokenized and embedded.
- **Model**: A machine learning classifier (e.g., Random Forest, SVM, or Neural Network) predicts the color outcome based on input features.
- **Output**: Color label (e.g., 'blue', 'green', 'brown') and optional confidence score.

Model retraining scripts are provided in the `model/` directory.

---

## Web Application

The web interface is built using Flask, providing:
- Simple forms for input
- Real-time prediction display
- Easy integration with backend model

---

## Contributing

Contributions are welcome! To contribute:
- Fork the repository
- Create a new branch for your feature/fix
- Submit a pull request

---

## License

This project is licensed under the MIT License.

---

## Acknowledgements

- Developed as part of the NFSU Forensics Hackathon 2023, where it won ₹2 lakh seed funding.
- Inspired by advances in AI for chemical engineering and forensic chemistry.
- Thanks to the open-source community for foundational libraries and datasets.

---

**Empowering forensic science with AI-driven chemical analysis.**
