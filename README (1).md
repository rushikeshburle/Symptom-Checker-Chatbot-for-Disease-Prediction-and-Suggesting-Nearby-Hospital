
# Project Title

 Development of Machine Learning  based Symptom Checker Chatbot for Disease Prediction and Suggesting Nearby hospital


## Introduction
A simple chatbot that can help users check their symptoms and provide relevant information. This chatbot uses natural language processing (NLP) techniques and a recurrent neural network (RNN) model to understand user queries and respond accordingly.


## Roadmap

To get started with the Symptom Checker Chatbot, follow these steps:

Clone the repository to your local machine:

git clone https://github.com/your-username/symptom-checker-chatbot.git
Install the required dependencies:
pip install -r requirements.txt

Download the necessary data files and place them in the appropriate directories:

intents.json: Contains predefined intents for the chatbot.

data_rnn.pth: Pre-trained RNN model for natural language understanding.

medical_centers.json: Data file containing information about medical centers.

## Usage

To use the Symptom Checker Chatbot, run the app.py file:

python app.py
This will start a Flask web application that you can interact with. Visit the URL http://localhost:5000 in your web browser to access the chat interface.

Training
If you want to retrain the RNN model or modify the chatbot's behavior, follow these steps:

Modify the intents and patterns in intents.json to customize the chatbot's responses.

Run the training script to retrain the RNN model:

python train.py

Save the trained model with a new filename and update the FILE variable in chat.py with the new filename.



## Installation

Install my-project 
The Symptom Checker Chatbot relies on the following dependencies:

Python 3.x

PyTorch

Flask

NLTK

geocoder

You can install these dependencies using the provided file.
## Contributing

Contributions are always welcome!

If you'd like to contribute to this project, feel free to fork the repository and submit pull requests. Contributions are welcome, whether it's bug fixes, feature enhancements, or documentation improvements.


## Contact
For questions or collaboration opportunities, feel free to contact us at rushikeshburle@gmail.com.