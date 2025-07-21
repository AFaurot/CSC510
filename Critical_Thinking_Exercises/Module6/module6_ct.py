import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder

'''README:
This script trains a Naive Bayes classifier to classify SMS messages as spam or not spam (ham).
The dataset used for training is the SMS Spam Collection Dataset, which contains labeled messages.
For this script to work, you need to download the dataset file from the UCI Machine Learning Repository.
https://archive.ics.uci.edu/dataset/228/sms+spam+collection
Rename the file to "spam.csv" and place it in the same directory as this script. 

References: 
Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection [Dataset]. 
     UCI Machine Learning Repository. https://doi.org/10.24432/C5CC84.
'''


def main():

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer(stop_words='english', lowercase=True)

    # Load the dataset using tab separator -
    # latin-1 encoding handles special characters - see discussion on Kaggle
    # https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset/discussion/346758
    df = pd.read_csv("spam.csv", sep='\t', encoding='latin-1', header=None)

    # Rename columns
    df.columns = ['label', 'message']

    # Encode string labels ("ham", "spam") into integers (0, 1)
    encoder = LabelEncoder()
    y = encoder.fit_transform(df['label'])

    # Convert messages to a frequency table using CountVectorizer
    X = vectorizer.fit_transform(df['message'])
    # Alpha is set to 1.0 for Laplace smoothing
    model = MultinomialNB(alpha=1.0)
    # Fit the model to the training data,
    model.fit(X, y)

    # Testing loop for user input
    while True:
        test_phrase = input("Enter a test phrase to classify as spam or not spam: ")
        test_vector = vectorizer.transform([test_phrase])
        # Predict the class of the test
        prediction = model.predict(test_vector)
        # Get the posterior probabilities for each class
        proba = model.predict_proba(test_vector)
        print(f"The phrase '{test_phrase}' is classified as: {encoder.inverse_transform(prediction)[0]}")
        for label, prob in zip(encoder.classes_, proba[0]):
            print(f"Probability of '{label}': {prob:.4f}")
        test_again = input("Do you want to test another phrase? (y to continue): ")
        if test_again.lower() != 'y':
            break


if __name__ == "__main__":
    main()


