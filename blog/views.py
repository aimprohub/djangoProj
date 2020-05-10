from django.shortcuts import render
from django.http import HttpResponse
from .models import Post
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import json
import requests
# data
# posts = [
#     {
#         'author': 'Shrikant K.',
#         'title': 'Django UI Framework',
#         'content': 'Learn how to make UI using Django',
#         'date_posted': 'August 27, 2018'
#     },
#     {
#         'author': 'Girish M.',
#         'title': 'Data Science',
#         'content': 'Learn Data Science Fundamentals.',
#         'date_posted': 'August 28, 2018'
#     }
# ]

# Create your views here.

@login_required()
def home(request):
    context = {'posts': Post.objects.all().order_by('-date_posted'), 'title': 'Home'}
    return render(request, 'blog/home.html', context)

@login_required()
def about(request):
    return render(request, 'blog/about.html', {'title': 'About'})
@login_required()
def prediction(request): 
    return render(request, 'blog/prediction.html', {'title':'Prediction'})

@login_required()
def upload(request): 
    return render(request, 'blog/upload.html', {'title':'Upload'})

@login_required()
def name(request): 
    return render(request, 'blog/name.html', {'title':'Name'})

@login_required()
def name2(request): 
    return render(request, 'blog/name2.html', {'title':'Name2'})

@login_required()
def trial(request):
    if request.method=='POST':
        income=request.POST.get('income')
        age=request.POST.get('age')
        income = int(income)
        age = int(age)
        dist=request.POST.get('dist')
        exp=request.POST.get('exp')

        add=income+age

        print(add)

        return render(request, 'blog/name.html', {"addition" : add} )
    return render(request, 'blog/prediction.html')

@login_required()
def trial1(request):
    import pandas as pd
    import numpy as np


    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    from sklearn import preprocessing
    if request.method=='POST':
        par1=float(request.POST.get('par1'))
        par2=float(request.POST.get('par2'))
        par3=float(request.POST.get('par3'))
        par4=float(request.POST.get('par4'))
        par5=float(request.POST.get('par5'))
        par6=float(request.POST.get('par6'))
        par7=float(request.POST.get('par7'))
        par8=float(request.POST.get('par8'))
    par1=float(request.POST.get('par1'))
    par2=float(request.POST.get('par2'))
    par3=float(request.POST.get('par3'))
    par4=float(request.POST.get('par4'))
    par5=float(request.POST.get('par5'))
    par6=float(request.POST.get('par6'))
    par7=float(request.POST.get('par7'))
    par8=float(request.POST.get('par8'))
     

    data = pd.read_csv('C:/Users/tanma/.spyder-py3/project.csv')

    data['Rate'] = data['DailyRate'] * 20 + data['HourlyRate'] * 8 * 20 + data['MonthlyRate']

    data.drop(['Department','EducationField'],axis=1).head()

    sex = pd.get_dummies(data['Gender'],drop_first=True)

    BT = pd.get_dummies(data['BusinessTravel'],drop_first=True)
    BT=BT.drop(['Travel_Rarely'],axis=1)

    Att = pd.get_dummies(data['Attrition'],drop_first=True)
    Att = Att.rename(columns={"Yes":"Attr"})

    MS = pd.get_dummies(data['MaritalStatus'],drop_first=True)
    MS = MS.drop(['Single'],axis=1)

    ovt = pd.get_dummies(data['OverTime'],drop_first=True)
    ovt = ovt.rename(columns={"Yes":"Ovt"})

    ne=data.drop(['Attrition','Gender','BusinessTravel','Department','EducationField','OverTime','Over18','JobRole','MaritalStatus','EmployeeNumber'],axis=1)

    daa = pd.concat([ne,sex,BT,Att,MS,ovt],axis=1)

    laa=daa.drop(['MonthlyRate','DailyRate','HourlyRate'],axis=1)

    data = daa[['Attr','MonthlyIncome','Rate', 'Age', 'Ovt', 'TotalWorkingYears', 'YearsAtCompany','YearsInCurrentRole','DistanceFromHome']]

    X = data.iloc[:, 1:10]
    y = data.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 120)

    sc = StandardScaler()
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    logmodel = LogisticRegression()
    logmodel.fit(X_train,y_train)

    y_pred = logmodel.predict(X_test)

    #new_data = [4736,32589,30,1,4,2,2,25]
    new_data = [par1,par2,par3,par4,par5,par6,par7,par8]
    input1 =  scaler.transform([new_data])
    
    result = int(logmodel.predict(input1))
    #print(result)

    return render(request, 'blog/name.html', {"addition" : result, "par1" : par1, "par2" : par2, "par3" : par3,
    "par4" : par4, "par5" : par5, "par6" : par6, "par7" : par7} )
    
@login_required()
def read(request):
    
    if request.method =='POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()   
        fs.save(uploaded_file.name, uploaded_file) 
    

        #latest_file = max(list_of_files, key = os.path.getctime)
        #print(latest_file)
        #return render(request, 'blog/name2.html')
    return render(request, 'blog/name2.html')    
