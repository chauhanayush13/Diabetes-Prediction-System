from django.shortcuts import render
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def home(request):
    return render(request , 'home.html')

def performance(request):
    return render(request , 'performance.html')

def predict(request):
    return render(request , 'predict.html')

def about(request):
    return render(request , 'about.html')

def result(request):

    dataset = pd.read_csv("C:/Users/amc13/Downloads/mini project/diabetes_prediction_dataset1.csv")
    
    x = dataset.drop('diabetes' , axis = 1)
    y = dataset['diabetes']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    
    val1 = float(request.GET.get('gender'))
    val2 = float(request.GET.get('age'))
    val3 = float(request.GET.get('height'))
    val4 = float(request.GET.get('weight'))
    val5 = float(request.GET.get('HbA1c_level'))
    val6 = float(request.GET.get('glucose'))
    
    #bmi
    val7 = val4/val3;
    pred = model.predict([[val1, val2, val7 , val5 , val6]])

    result1 = " "
    if pred == [1]:
        result1 = "Positive"
    else:
        result1 = "Negative"

    return render(request, 'predict.html', {"result2": result1})
    

