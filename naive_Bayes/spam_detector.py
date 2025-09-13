import pandas as pd
from naive_Bayes import NaiveBayes

data_classes = pd.read_csv("letter_spam_freq.csv")
data_cases = pd.read_csv("words_spam_freq.csv")

model = NaiveBayes(data_classes, data_cases)
print(model)

print(model.predict(["Online", "Million", "Access", "Cash", "Bill", "Offer", "Money"]))
