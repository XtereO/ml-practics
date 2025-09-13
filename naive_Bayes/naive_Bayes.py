from functools import reduce
from math import log, exp

class NaiveBayes:
    def __init__(self, data_classes, data_cases):
        self.classes = data_classes.columns
        self.classes_freq = data_classes.iloc[:]
        self.classes_cases = {_class: data_cases[(data_cases[_class]>0)].iloc[:, 0] for _class in self.classes}
        self.classes_cases_cardinalities = {_class: len(self.classes_cases[_class]) for _class in self.classes_cases}

        classes_freq_sum = self.classes_freq.iloc[0].sum()
        self.classes_probability = {c: (self.classes_freq.loc[0, c]/classes_freq_sum) for c in self.classes}

        self.cases = data_cases.iloc[:, 0]
        self.cases_freq = data_cases.iloc[:]
        self.classes_cases_freq = {_class: self.cases_freq[_class].sum() for _class in self.classes}

    def predict(self, cases):
        exp_probabilities = {_class: 0 for _class in self.classes}
        for _class in self.classes:
            r = len(set(cases) - set(self.classes_cases[_class]))
            denominator = (self.classes_cases_cardinalities[_class]+r+self.classes_cases_freq[_class])
            ln_p_class = log(self.classes_probability[_class])
            for c in cases:
                case_freq = 0
                try:
                    # try to get case freq for specific class
                    case_freq = self.cases_freq.loc[self.cases_freq.iloc[:, 0]==c, _class].iloc[0]
                except Exception as e:
                    print("the case:", c, e)
                    case_freq = 0 
                finally:
                    ln_p_class+=log((case_freq+1)/denominator)
 
            exp_probabilities[_class] = exp(ln_p_class)

        sum_exp_probabilities = reduce(lambda s, ln_p: s+ln_p, exp_probabilities.values(), 0)
        return {_class: exp_probabilities[_class]/sum_exp_probabilities for _class in self.classes}     

    def __str__(self):
        return f"classes:\n{self.classes_freq}\n\ncases:\n{self.cases_freq}\n\nclasses cases:\n{self.classes_cases}\n\nclasses probability:\n{self.classes_probability}"
    