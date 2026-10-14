# AI Image Recognition System

## Project Overview
The AI Image Recognition System is a Python-based application that uses a pre-trained Vision Transformer (ViT) model to identify objects and categories in images. Users can upload an image through a Streamlit web interface and view the model's top predictions and scores.

## Features
- Upload JPG, JPEG, and PNG images.
- Display the uploaded image.
- Predict image categories using a pre-trained AI model.
- Show the top five predictions and their scores.
- Visualize prediction scores with a bar chart.
- Run image classification from the command line.

## Technologies Used
- Python
- Streamlit
- Hugging Face Transformers
- PyTorch
- Pillow
- Pandas
- Plotly

## AI Model
This project uses the pre-trained `google/vit-base-patch16-224` model.

The model performs image classification by comparing image features with categories it learned during training.

## Project Structure
```text
Image-Recognition-System/
├── app.py
├── image_classifier.py
├── test_model.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Create a virtual environment:

   `python -m venv .venv`

4. Activate the environment on Windows:

   `.venv\Scripts\activate`

5. Install the dependencies:

   `pip install -r requirements.txt`

## Run the Web Application

Run this command:

`python -m streamlit run app.py`

Open the local URL shown in the terminal, upload an image, and view the predictions.

## Run Image Classification from the Terminal

Run:

`python image_classifier.py`

Enter the path to an image when prompted.

## Run the Test

Make sure `test.jpg` is available in the project directory, then run:

`python test_model.py`

The test checks that predictions are returned and include a label and score.

## Limitations
- Predictions depend on image quality and the categories supported by the model.
- The model may misclassify unusual or unclear images.
- Scores should not be treated as guaranteed probabilities of correctness.

## Future Improvements
- Support more image formats.
- Add image history and downloadable results.
- Improve the user interface.
- Deploy the application online.

## Author
Vilashini Ramesh
