import pandas as pd
import numpy as np
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.sparse import hstack
from sklearn.preprocessing import LabelEncoder
from joblib import load
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove punctuation and special characters using regex
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove stop words
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

df=pd.read_csv('emotion_data.csv')
# TF-IDF with n-grams
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3))
X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

# Bag of Words with n-grams
count_vectorizer = CountVectorizer(ngram_range=(1, 3))
X_bow = count_vectorizer.fit_transform(df['processed_text'])

# Concatenate features
X_combined = hstack([X_tfidf, X_bow])

# Target variable
y = df['label']
le=LabelEncoder()
y=le.fit_transform(y)

# from xgboost import XGBClassifier
# import seaborn as sns
# my_model=XGBClassifier()
# my_model.fit(Xtrain,Ytrain, verbose=False)
# predictions = my_model.predict(Xtest)
# from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
# print("ACCURACY SCORE IS : "+str(accuracy_score(Ytest,predictions)))

# To load the model later
loaded_model = load('svc_model.joblib')

def preprocess_and_predict(texts,model):
    # Vectorize text
    X_tfidf = tfidf_vectorizer.transform(texts)
    X_bow = count_vectorizer.transform(texts)
    # Concatenate features
    X_combined = hstack([X_tfidf, X_bow])
    
    return model.predict_proba(X_combined)

def explain(text, loaded_model=loaded_model):
    from lime.lime_text import LimeTextExplainer
    import matplotlib.pyplot as plt
    from IPython.core.display import display, HTML

    # Initialize LIME Text Explainer
    explainer = LimeTextExplainer(class_names=le.classes_)

    # Preprocess the input text
    text = preprocess_text(text)
    text_to_explain = text

    # Explain the prediction
    explanation = explainer.explain_instance(
        text_to_explain,
        lambda x: preprocess_and_predict(x, loaded_model),
        num_features=6
    )

    # Save explanation as an image with proper figure size
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(10, 8)  # Adjust the figure size as needed
    image_path = "lime.png"
    fig.savefig(image_path, bbox_inches='tight')  # Ensure everything fits
    plt.close(fig)  # Close the figure to free memory

    # print(f"Explanation saved as image: {image_path}")

    # # Save explanation as HTML with UTF-8 encoding
    # html_path = 'C:/Users/Subhasrikar/Documents/angular/fast-api_backend/lime.html'
    # with open(html_path, "w", encoding="utf-8") as f:
    #     f.write(explanation.as_html())
    # print(f"Explanation saved as HTML: {html_path}")

    # Optionally display the HTML in Jupyter Notebook
    # display(HTML(explanation.as_html()))

    return image_path


def predic(text,model=loaded_model):
    text=preprocess_text(text)
    predi=preprocess_and_predict([text],loaded_model)
    maxone= np.argmax(predi)
    return le.inverse_transform([maxone])[0]

# tex=input("enter text")

# print(predic(tex,loaded_model))
# print(explain(tex,loaded_model))
