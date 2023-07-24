
# Image text (OCR) reader and Text Analyzer
**Author:** [@harshpx](https://github.com/harshpx)
## Overview
A Machine learning OCR and Sentiment Analysis model.
Tech Stack: Python, Scikit-Learn, Easy OCR, Streamlit.

#### To see this app working live: 
Hop onto: 


#### To run this project on your local system: 
To deploy this project locally, first make sure you have all dependencies installed. *see ```requirements.txt```*

And Run: ```streamlit app.py```

or ```python3 -m streamlit app.py``` 

in the terminal of file directory.


## Brief Project Description
1. **Text Extraction from Image**
* Used EasyOCR API for text extraction. EasyOCR is an open-source Optical Character Recognition (OCR) engine made using PyTorch. 

2. **Emotion Detection**
* It is a Multiclass classification Model that uses various small datasets of texts and labelled emotion from Kaggle. 
* The overall combined dataset consists of approx 30,000 labelled texts, that spans accross 9 emotions. Namely: `anger`, `disgust`, `fear`, `guilt`, `joy`, `love`, `sadness`, `shame` and `surprise`. (A 9 class classification problem)
* Various Machine Learning classification Models like SVM, Logistic Regression, Ensemble methods etc. are applied.

Read More about Stacking Ensemble: [Stacking Ensemble-Machine Learning Mastery](https://machinelearningmastery.com/stacking-ensemble-machine-learning-with-python/)
and Support Vector Machines(SVM): [SVMs-Towards Data Science](https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47)
        

3. **Sentiment Detection**
* It is also a Multiclass classification Model that uses a modified dataset derived from `Sentiment 140`, that has 50k tweets spanning accross 3 sentiments. Namely: `Positive`, `Neutral` and `Negative`. (A 3 class classification problem)
* Various Machine Learning Classifiers like: `Logistic Regression`, `SVM`, `Decision Trees`, `Naive Bayes`, `KNN` etc. are tested during Hyperparameter Tuning (using `GridSearchCV()`)
`Logistic Regression` performed best among them and achieved an accuray of **82%**.

Read More about Logistic Regression: [Logistic Regression-Towards Data Science](https://towardsdatascience.com/logistic-regression-detailed-overview-46c4da4303bc)

## Source Codes
* [Kaggle: BasicOCR](https://www.kaggle.com/code/harshpriye/ocr-text-analysis/notebook)
* [Kaggle: Emotion-detection-from-texts](https://www.kaggle.com/code/harshpriye/emotion-detection-from-texts)
* [Kaggle: Sentiment-analysis-from-texts](https://www.kaggle.com/code/harshpriye/sentiment-analysis-from-texts)
