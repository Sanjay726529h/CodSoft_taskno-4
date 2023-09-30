**Spam SMS Detection**
This project focuses on detecting spam SMS messages using natural language processing techniques. The task involves classifying SMS messages as either "ham" (non-spam) or "spam." Below is an overview of the key steps and components of this project:

**Dataset**
The dataset used for this task is provided in the file "spam.csv," encoded in Latin-1.
**Data Preprocessing**
The dataset is loaded and examined to understand its structure and contents.
Unnecessary columns, such as "Unnamed: 2," "Unnamed: 3," and "Unnamed: 4," are dropped from the dataset.
The columns are renamed to more descriptive names: "label" for the message label and "text" for the SMS text content.
Labels are encoded, where "ham" is mapped to 0, and "spam" is mapped to 1.
The distribution of "ham" and "spam" messages is visualized using a count plot.
**Text Analysis**
The average number of tokens (words) in the SMS messages is calculated.
The total number of unique words in the corpus is determined.
**Data Splitting**
The dataset is split into training and testing sets using a 70-30 split.
**Feature Extraction**
Text data is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.
The TF-IDF vectorizer is fitted on the training data and used to transform both the training and testing sets.
**Model Building**
A Multinomial Naive Bayes classifier is trained on the TF-IDF-transformed training data to classify SMS messages as "ham" or "spam."
**Model Evaluation**
The model's accuracy, precision, recall, F1 score, and classification report are generated using the testing data.
A confusion matrix is created to visualize the model's performance.
**Model Serialization**
The trained Multinomial Naive Bayes model is serialized and saved to a file named "spam_sms_detection_model.pkl."
The TF-IDF vectorizer used for feature extraction is also serialized and saved to a file named "tfidf_vectorizer.pkl."
**Model Deployment (Optional)**
The deployed model can be used to classify new SMS messages as "ham" or "spam" by inputting a message text.
Feel free to explore the code in the provided Jupyter Notebook to gain a more in-depth understanding of the project. If you have any questions or need further information, please don't hesitate to reach out.

Best regards,

Sanjay Kumar
