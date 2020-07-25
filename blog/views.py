from django.shortcuts import render
from django.http import HttpResponse
from .models import Post
from django.contrib import messages
from django.core.files.storage import FileSystemStorage
from django.contrib.auth.decorators import login_required
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import json
import requests
import os.path
from os import path


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
def stats(request): 
    return render(request, 'blog/stats.html', {'title':'stats'})

@login_required()
def upload1(request): 
    return render(request, 'blog/upload1.html', {'title':'upload1'})

@login_required()
def upload2(request): 
    return render(request, 'blog/upload2.html', {'title':'upload2'})


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
    from sklearn.impute import SimpleImputer 

    from sklearn.model_selection import train_test_split


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
        par9=float(request.POST.get('par9'))

    par1=float(request.POST.get('par1'))
    par2=float(request.POST.get('par2'))
    par3=float(request.POST.get('par3'))
    par4=float(request.POST.get('par4'))
    par5=float(request.POST.get('par5'))
    par6=float(request.POST.get('par6'))
    par7=float(request.POST.get('par7'))
    par8=float(request.POST.get('par8'))
    par9=float(request.POST.get('par9'))
    

    data = pd.read_csv('C:/Users/tanma/.spyder-py3/project_new.csv')
    new_data = data.drop(['Department','EducationField','EmployeeCount','EmployeeNumber','Application ID','JobRole','Over18','Employee Source'],axis=1)


################################################################
    attr = pd.get_dummies(new_data['Attrition'],drop_first=True)
    attr = attr.rename(columns={"Voluntary Resignation":"Attr"})
    attr=attr.drop(['Termination'],axis=1)

    sex = pd.get_dummies(new_data['Gender'],drop_first=True)
    sex = sex.rename(columns={"Female":"Gender"})  #female1 Male0
    sex = sex.drop(['Male'],axis=1)
    sex = sex.drop(['2'],axis=1)

    bt = pd.get_dummies(new_data['BusinessTravel'],drop_first=True)
    bt=bt.drop(['Travel_Rarely'],axis=1)

    ms = pd.get_dummies(new_data['MaritalStatus'],drop_first=True)  
    ms = ms.drop(['Single','Divorced'],axis=1) 

    ovt = pd.get_dummies(new_data['OverTime'],drop_first=True)
    ovt = ovt.drop(['Y'],axis=1) 
    ovt = ovt.rename(columns={"Yes":"Ovt"})

    ################################################################

    ne=new_data.drop(['Attrition','Gender','BusinessTravel','OverTime','MaritalStatus'],axis=1)

    daa = pd.concat([ne,sex,bt,attr,ms,ovt],axis=1)
    ########################################################################################

    daa['dfh'] = pd.to_numeric(daa.DistanceFromHome.astype(str).str.replace(',',''), errors='coerce')
    daa=daa.drop(['DistanceFromHome'],axis=1)

    daa['hourlyr'] = pd.to_numeric(daa.HourlyRate.astype(str).str.replace(',',''), errors='coerce')
    daa=daa.drop(['HourlyRate'],axis=1)

    daa['Jobs'] = pd.to_numeric(daa.JobSatisfaction.astype(str).str.replace(',',''), errors='coerce')
    daa=daa.drop(['JobSatisfaction'],axis=1)

    daa['MonthlyInc'] = pd.to_numeric(daa.MonthlyIncome.astype(str).str.replace(',',''), errors='coerce')
    daa=daa.drop(['MonthlyIncome'],axis=1)


    daa['PercentSalHike'] = pd.to_numeric(daa.PercentSalaryHike.astype(str).str.replace(',',''), errors='coerce')
    daa=daa.drop(['PercentSalaryHike'],axis=1)

    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(daa)
    daa2 = imp.transform(daa)

    for i, col in enumerate(daa.columns):
        daa[col] = daa2[:,i]
        

    new =daa[['Attr','TotalWorkingYears','Ovt','YearsAtCompany','YearsInCurrentRole','YearsWithCurrManager','dfh','Age','MonthlyInc','EnvironmentSatisfaction','NumCompaniesWorked']]

    X = new.iloc[:, 1:10]
    y = new.iloc[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 45)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=110)
    rfc.fit(X_train,y_train)
    y_pred = rfc.predict(X_test)

    new_data = [par5,par4,par6,par7,par2,par8,par3,par1,par9]

    input1 =  scaler.transform([new_data])
    
    predict = int(rfc.predict(input1))
    #print(predict)
    probs = rfc.predict_proba(input1)[:,1]
    perc = int(probs*100)
    #print(perc)

    #image
    

    return render(request, 'blog/name.html', {"addition" : predict, "par1" : par1, "par2" : par2, "par3" : par3,
    "par4" : par4, "par5" : par5, "par6" : par6, "par7" : par7, "perc" : perc} )
    
@login_required()
def read(request):
    import os
    if path.exists(r'C:\Users\tanma\old_myproj\djangoProj\media\single_file_upload.csv'):
        os.remove(r'C:\Users\tanma\old_myproj\djangoProj\media\single_file_upload.csv')
    if request.method =='POST':
        uploaded_file = request.FILES['document']
        fs = FileSystemStorage()   
        fs.save('single_file_upload.csv', uploaded_file)

    if path.exists(r'C:\Users\tanma\old_myproj\djangoProj\media\single_file_upload.csv'):   
        
        import pandas as pd
        import numpy as np

        from sklearn import preprocessing
        from sklearn.impute import SimpleImputer 

                #from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        #from sklearn.linear_model import LogisticRegression
        data = pd.read_csv('C:/Users/tanma/.spyder-py3/IBM_train.csv')
        new_data = data.drop(['Department','EducationField','EmployeeCount','EmployeeNumber','Application ID','JobRole','Over18','Employee Source'],axis=1)


        ################################################################

        sex = pd.get_dummies(new_data['Gender'],drop_first=True)
        sex = sex.rename(columns={"Female":"Gender"})  #female1 Male0
        sex = sex.drop(['Male'],axis=1)
        sex = sex.drop(['2'],axis=1)

        bt = pd.get_dummies(new_data['BusinessTravel'],drop_first=True)
        bt=bt.drop(['Travel_Rarely'],axis=1)

        ms = pd.get_dummies(new_data['MaritalStatus'],drop_first=True)  
        ms = ms.drop(['Single','Divorced'],axis=1) 

        ovt = pd.get_dummies(new_data['OverTime'],drop_first=True)
        ovt = ovt.drop(['Y'],axis=1) 
        ovt = ovt.rename(columns={"Yes":"Ovt"})

        ################################################################

        ne=new_data.drop(['Gender','BusinessTravel','OverTime','MaritalStatus'],axis=1)

        daa = pd.concat([ne,sex,bt,ms,ovt],axis=1)
        ########################################################################################

        daa['dfh'] = pd.to_numeric(daa.DistanceFromHome.astype(str).str.replace(',',''), errors='coerce')
        daa=daa.drop(['DistanceFromHome'],axis=1)

        daa['hourlyr'] = pd.to_numeric(daa.HourlyRate.astype(str).str.replace(',',''), errors='coerce')
        daa=daa.drop(['HourlyRate'],axis=1)

        daa['Jobs'] = pd.to_numeric(daa.JobSatisfaction.astype(str).str.replace(',',''), errors='coerce')
        daa=daa.drop(['JobSatisfaction'],axis=1)

        daa['MonthlyInc'] = pd.to_numeric(daa.MonthlyIncome.astype(str).str.replace(',',''), errors='coerce')
        daa=daa.drop(['MonthlyIncome'],axis=1)


        daa['PercentSalHike'] = pd.to_numeric(daa.PercentSalaryHike.astype(str).str.replace(',',''), errors='coerce')
        daa=daa.drop(['PercentSalaryHike'],axis=1)

        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(daa)
        daa2 = imp.transform(daa)

        for i, col in enumerate(daa.columns):
            daa[col] = daa2[:,i]


        new =daa[['Attr','TotalWorkingYears','Ovt','YearsAtCompany','YearsInCurrentRole','YearsWithCurrManager','dfh','Age','MonthlyInc','EnvironmentSatisfaction','NumCompaniesWorked']]

        X = new.iloc[:, 1:10]
        y = new.iloc[:, 0]
        test1 = pd.read_csv(r'C:\Users\tanma\old_myproj\djangoProj\media\single_file_upload.csv')
        print(test1)

        sc = StandardScaler()
        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        test2 = test1
        test1 = scaler.transform(test1)
        print("Printing TYpe of test2 ------", type(test2))
        print(test2)
                # print(type(test2))

                #test2=(test2.values.tolist())
                # print(type(x))

        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier(n_estimators=110)
        rfc.fit(X,y)
        y_pred = rfc.predict(test1)
        probs = rfc.predict_proba(test1)[:,1]
        probs = probs*100
        probs = np.around(probs, decimals = 2)  
            
        #logmodel = LogisticRegression()
        #logmodel.fit(X,y)
        #y_pred = rfc.predict(test1)

        #y_pred = logmodel.predict(test1)
        prob_with_perc = []
        for i in probs:
            prob_with_perc.append(str(i)+"%")      

        print(y_pred)
        print(test1)
        test2['probs'] = prob_with_perc
        test2['pred'] = y_pred
        test2['pred'] = test2['pred'].replace(1,"Yes").replace(0,"No")
        test2['Ovt'] = test2['Ovt'].replace(1,"Yes").replace(0,"No")
        test2['EnvSatisfaction'] = test2['EnvSatisfaction'].replace(1,"Poor").replace(2,"Average").replace(3,"Good").replace(4,"Very Good").replace(5,"Excellent")


        print(test2)
                #test2['Ovt'] = test2['Ovt'].replace(1,"Yes").replace(0,"No")
        test2 = test2.values.tolist()
                #print(test2)
                #test2.insert(8,y_pred)
                #test2['pred'] = y_pred
        print(test2)
        
        
    else:
        messages.error(request,'Please Upload CSV File')
        #latest_file = max(list_of_files, key = os.path.getctime)
        #print(latest_file)
        #return render(request, 'blog/name2.html')
    #os.remove(r'C:\Users\tanma\old_myproj\djangoProj\media\project.csv')

    model_name = "Random Forest Classifier"
    a = 2
    a_json = json.dumps(a)

    return render(request, 'blog/name2.html',{"test2" : test2, "y_pred"  : y_pred,"model_name"  : model_name,"a" : a_json,})    


@login_required()
def multi_read(request):
    import os
    if path.exists(r'C:\Users\tanma\old_myproj\djangoProj\media\user_training.csv'):
        os.remove(r'C:\Users\tanma\old_myproj\djangoProj\media\user_training.csv')
    if path.exists(r'C:\Users\tanma\old_myproj\djangoProj\media\user_test.csv'):
        os.remove(r'C:\Users\tanma\old_myproj\djangoProj\media\user_test.csv')
    if request.method =='POST':
        uploaded_file = request.FILES['document1']
        fs = FileSystemStorage()   
        fs.save('user_training.csv', uploaded_file)
        uploaded_file2 = request.FILES['document2']
        fs = FileSystemStorage()   
        fs.save('user_test.csv', uploaded_file2)
        import pandas as pd
        #from sklearn.linear_model import LogisticRegression
        from sklearn.impute import SimpleImputer 

        from sklearn.metrics import confusion_matrix


        from sklearn import preprocessing
        import numpy as np

        #data = pd.read_csv('C:/Users/tanma/.spyder-py3/user_train.csv')
        data = pd.read_csv(r'C:\Users\tanma\old_myproj\djangoProj\media\user_training.csv')
        data=data.drop(['NumCompaniesWorked'],axis=1)

        from sklearn.model_selection import train_test_split

        X = data.iloc[:, 1:10]
        y = data.iloc[:, 0]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 155)

        scaler = preprocessing.StandardScaler().fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)


        def train(model):
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            cm = confusion_matrix(y_test,y_pred)
            accuracy = round(100*np.trace(cm)/np.sum(cm),1)
            return accuracy



        def lr():
            from sklearn.linear_model import LogisticRegression
            lrmodel = LogisticRegression()
            return train(lrmodel)

        def svc():
            from sklearn.svm import SVC
            svcmodel = SVC(kernel='rbf') 
            return train(svcmodel)

        # def xgb():
        #     from xgboost import XGBClassifier
        #     xgbmodel = XGBClassifier()
        #     return train(xgbmodel)


        def dt():
            from sklearn.tree import DecisionTreeClassifier
            dtmodel = DecisionTreeClassifier()
            return train(dtmodel)

        def knn():
            from sklearn.neighbors import KNeighborsClassifier
            knnmodel = KNeighborsClassifier(n_neighbors=27)
            return train(knnmodel)

        def rf():
            from sklearn.ensemble import RandomForestClassifier
            rfmodel = RandomForestClassifier(n_estimators=110)
            return train(rfmodel)

        def nb():
            from sklearn.naive_bayes import GaussianNB
            nbmodel = GaussianNB()
            return train(nbmodel)

        # def ann():
        #     dropout = 0.1
        #     epochs = 100
        #     batch_size = 30
        #     optimizer = 'adam'
        #     k = 20
        #     from keras.wrappers.scikit_learn import KerasClassifier
        #     from keras.models import Sequential
        #     from keras.layers import Dense
        #     from keras.layers import Dropout
        #     def build_classifier():
        #         classifier = Sequential()
        #         classifier.add(Dense(16, kernel_initializer="truncated_normal", activation = 'relu', input_shape = (X.shape[1],)))
        #         classifier.add(Dropout(dropout))
        #         classifier.add(Dense(1, kernel_initializer="truncated_normal", activation = 'sigmoid', )) #outputlayer
        #         classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ["accuracy"])
        #         return classifier
        #     classifier = KerasClassifier(build_fn = build_classifier, batch_size = batch_size, epochs = epochs, verbose=0)
        #     classifier.fit(X_train,y_train)
        #     y_pred=classifier.predict(X_test)
        #     y_pred =(y_pred>0.5)
        #     cm = confusion_matrix(y_test,y_pred)
        #     accuracy = round(100*np.trace(cm)/np.sum(cm),1)
        #     return accuracy


        def max_accuracy(lracc,svcacc,dtacc,knnacc,rfacc,nbacc):
            #return max(lracc,svcacc,xgbacc,dtacc,knnacc,rfacc,nbacc,annacc)
            return max(lracc,svcacc,dtacc,knnacc,rfacc,nbacc)

        lracc = lr()
        svcacc = svc()
        #xgbacc = xgb()
        dtacc = dt()
        knnacc = knn()
        rfacc = rf()
        nbacc = nb()
        #annacc = ann()
            
        #maxx = max_accuracy(lracc,svcacc,xgbacc,dtacc,knnacc,rfacc,nbacc,annacc)
        maxx = max_accuracy(lracc,svcacc,dtacc,knnacc,rfacc,nbacc)


        #test = pd.read_csv('C:/Users/tanma/.spyder-py3/new_test.csv')
        test = pd.read_csv(r'C:\Users\tanma\old_myproj\djangoProj\media\user_test.csv')
        print(test)
        test2=test

        scaler = preprocessing.StandardScaler().fit(X)
        X = scaler.transform(X)
        test1 = scaler.transform(test)
        print(test1)
        model_name=None
        if lracc == maxx:
            from sklearn.linear_model import LogisticRegression
            logmodel = LogisticRegression()
            logmodel.fit(X,y)
            model_name = "Logistic Regression"
            y_pred = logmodel.predict(test1)
            print(y_pred,',logistic')
            probs = logmodel.predict_proba(test1)[:,1]
            probs = probs*100
            
        elif svcacc == maxx:
            from sklearn.svm import SVC
            svcmodel = SVC(kernel='rbf')
            svcmodel.fit(X,y)
            model_name = "Support Vector Machine"
            y_pred = svcmodel.predict(test1)
            
            print(y_pred,'svc')
            probs = svcmodel.predict_proba(test1)[:,1]
            probs = probs*100
        # elif xgbacc == maxx:
        #     xgbmodel = XGBClassifier()
        #     xgbmodel.fit(X,y)
        #     y_pred = svcmodel.predict(test1)
            
        #     print(y_pred,'xgb')
            
        elif dtacc == maxx:
            from sklearn.tree import DecisionTreeClassifier
            dtmodel = DecisionTreeClassifier()
            dtmodel.fit(X,y)
            model_name = "Decision Tree Classifier"
            y_pred = dtmodel.predict(test1)
            
            print(y_pred,'dt')    
            probs = dtmodel.predict_proba(test1)[:,1]
            probs = probs*100        
            
        elif knnacc == maxx:
            from sklearn.neighbors import KNeighborsClassifier
            knnmodel = KNeighborsClassifier(n_neighbors=27)
            knnmodel.fit(X,y)
            model_name = "K Neighbors Classifier"
            y_pred = knnmodel.predict(test1)
            
            print(y_pred,'knn')
            probs = knnmodel.predict_proba(test1)[:,1]
            probs = probs*100
            
        elif rfacc == maxx:
            from sklearn.ensemble import RandomForestClassifier
            rfmodel = RandomForestClassifier(n_estimators=110)
            rfmodel.fit(X,y)
            model_name = "Random Forest Classifier"
            y_pred = rfmodel.predict(test1)
            
            print(y_pred,'rf')
            probs = rfmodel.predict_proba(test1)[:,1]
            probs = probs*100
            
        elif nbacc == maxx:
            from sklearn.naive_bayes import GaussianNB
            nbmodel = GaussianNB()
            nbmodel.fit(X,y)   
            model_name = "Naive Bayes" 
            y_pred = nbmodel.predict(test1)    
            print(y_pred,'nb')
            probs = nbmodel.predict_proba(test1)[:,1]
            probs = probs*100
        else:
            print('no')




        probs = np.around(probs, decimals = 2) 
        accuracy = {"Logistic":lracc,"SVM":svcacc,"DT":dtacc,"KNN":knnacc,"RF":rfacc,"NB":nbacc}
        print(accuracy)
        prob_with_perc = []
        for i in probs:
            prob_with_perc.append(str(i)+"%")

        print(y_pred)
        test2['probs'] = prob_with_perc
        test2['pred'] = y_pred
        test2['pred'] = test2['pred'].replace(1,"Yes").replace(0,"No")
        test2['Ovt'] = test2['Ovt'].replace(1,"Yes").replace(0,"No")
        test2['EnvSatisfaction'] = test2['EnvSatisfaction'].replace(1,"Poor").replace(2,"Average").replace(3,"Good").replace(4,"Very Good").replace(5,"Excellent")


        print(test2)
                        #test2['Ovt'] = test2['Ovt'].replace(1,"Yes").replace(0,"No")
        test2 = test2.values.tolist()
        #os.remove(r'C:\Users\tanma\old_myproj\djangoProj\media\user_training.csv')
        #os.remove(r'C:\Users\tanma\old_myproj\djangoProj\media\user_test.csv')




    return render(request, 'blog/name2.html',{"test2" : test2, "y_pred"  : y_pred,"model_name" : model_name,} )  

def stats2(request):
    import os
    if request.method =='POST':
        a = 2
        a_json = json.dumps(a)
        if path.exists(r'C:\Users\tanma\old_myproj\djangoProj\media\user_training.csv'):
            data = pd.read_csv(r'C:\Users\tanma\old_myproj\djangoProj\media\user_training.csv')
            #os.remove(r'C:\Users\tanma\old_myproj\djangoProj\media\user_training.csv')
        else:
            data = pd.read_csv('C:/Users/tanma/.spyder-py3/IBM_train.csv')
            
        #df1 = attr[1] df2 = attr[0]
        df1, df2 = [x for _, x in data.groupby(data['Attr'] < 1)]
        List_env_0=df2['EnvironmentSatisfaction'] #for attrition = 0
        env_0 = [0,0,0,0,0]
        for i in List_env_0:
            if i==1:
                env_0[0]+=1
            elif i==2:
                env_0[1]+=1
            elif i==3:
                env_0[2]+=1
            elif i==4:
                env_0[3]+=1
            elif i==5:
                env_0[4]+=1
        ENV1 = [['Satisfaction','Number of Employees'],['Poor',env_0[0]],['Average',env_0[1]],['Good',env_0[2]],['Very Good',env_0[3]],['Excellent',env_0[4]]]

        List_env_1=df1['EnvironmentSatisfaction'] #for attrition = 1
        env_1 = [0,0,0,0,0]
        for i in List_env_1:
            if i==1:
                env_1[0]+=1
            elif i==2:
                env_1[1]+=1
            elif i==3:
                env_1[2]+=1
            elif i==4:
                env_1[3]+=1
            elif i==5:
                env_1[4]+=1
        ENV2 = [['Satisfaction','Number of Employees'],['Poor',env_1[0]],['Average',env_1[1]],['Good',env_1[2]],['Very Good',env_1[3]],['Excellent',env_1[4]]]

        a_json = json.dumps(a)
        
        List_env_0_json = json.dumps(ENV1)
        List_env_1_json = json.dumps(ENV2)


        YrsinComp_0 = df2['Age']
        YrsinComp_1 = df1['Age']

        years_in_comp0 = [0,0,0,0,0,0,0]

        for i in YrsinComp_0:
            if i>20.0 and i<=25.0:
                years_in_comp0[0]+=1
            elif i>25 and i<=30:
                years_in_comp0[1]+=1
            elif i>30 and i<=35:
                years_in_comp0[2]+=1
            elif i>35 and i<=40:
                years_in_comp0[3]+=1
            elif i>40 and i<=45:
                years_in_comp0[4]+=1
            elif i>45 and i<=50:
                years_in_comp0[5]+=1
            else:
                years_in_comp0[6]+=1


            years_in_comp1 = [0,0,0,0,0,0,0]

        for i in YrsinComp_1:
            if i>20.0 and i<=25.0:
                years_in_comp1[0]+=1
            elif i>25 and i<=30:
                years_in_comp1[1]+=1
            elif i>30 and i<=35:
                years_in_comp1[2]+=1
            elif i>35 and i<=40:
                years_in_comp1[3]+=1
            elif i>40 and i<=45:
                years_in_comp1[4]+=1
            elif i>45 and i<=50:
                years_in_comp1[5]+=1
            else:
                years_in_comp1[6]+=1

        List_yrs_0 = [['Age Group','Employee Count'],['20-25',years_in_comp0[0]],['25-30',years_in_comp0[1]],['30-35',years_in_comp0[2]],['35-40',years_in_comp0[3]],['40-45',years_in_comp0[4]],['45-50',years_in_comp0[5]],['50 and above',years_in_comp0[6]]]
        List_yrs_1 = [['Age Group','Employee Count'],['20-25',years_in_comp1[0]],['25-30',years_in_comp1[1]],['30-35',years_in_comp1[2]],['35-40',years_in_comp1[3]],['40-45',years_in_comp1[4]],['45-50',years_in_comp1[5]],['50 and above',years_in_comp1[6]]]

        List_yrs_0_json = json.dumps(List_yrs_0)
        List_yrs_1_json = json.dumps(List_yrs_1)


        ##################################################################
        Comp_yrs_0 = df2['YearsAtCompany']
        Comp_yrs_1 = df1['YearsAtCompany']
        
        years_curr_comp0 = [0,0,0,0,0,0,0,0,0]

        for i in Comp_yrs_0:
            if i>0 and i<=2:
                years_curr_comp0[0]+=1
            elif i>2 and i<=4:
                years_curr_comp0[1]+=1
            elif i>4 and i<=6:
                years_curr_comp0[2]+=1
            elif i>6 and i<=8:
                years_curr_comp0[3]+=1
            elif i>8 and i<=10:
                years_curr_comp0[4]+=1
            elif i>10 and i<=12:
                years_curr_comp0[5]+=1
            elif i>12 and i<=14:
                years_curr_comp0[6]+=1
            elif i>14 and i<=16:
                years_curr_comp0[7]+=1
            else:
                years_curr_comp0[8]+=1
        
        years_curr_comp1 = [0,0,0,0,0,0,0,0,0]
        for i in Comp_yrs_1:
            if i>0 and i<=2:
                years_curr_comp1[0]+=1
            elif i>2 and i<=4:
                years_curr_comp1[1]+=1
            elif i>4 and i<=6:
                years_curr_comp1[2]+=1
            elif i>6 and i<=8:
                years_curr_comp1[3]+=1
            elif i>8 and i<=10:
                years_curr_comp1[4]+=1
            elif i>10 and i<=12:
                years_curr_comp1[5]+=1
            elif i>12 and i<=14:
                years_curr_comp1[6]+=1
            elif i>14 and i<=16:
                years_curr_comp1[7]+=1
            else:
                years_curr_comp1[8]+=1
            
        current_yrs_in_comp = [['Years in Company','-ve Attrition','+ve Attrition'],['0-2',years_curr_comp0[0],years_curr_comp1[0]],['2-4',years_curr_comp0[1],years_curr_comp1[1]],['4-6',years_curr_comp0[2],years_curr_comp1[2]],['6-8',years_curr_comp0[3],years_curr_comp1[3]],['8-10',years_curr_comp0[4],years_curr_comp1[4]],['10-12',years_curr_comp0[5],years_curr_comp1[5]],['12-14',years_curr_comp0[6],years_curr_comp1[6]],['14-16',years_curr_comp0[7],years_curr_comp1[7]],['16 and above',years_curr_comp0[8],years_curr_comp1[8]]]
        current_yrs_in_comp_json = json.dumps(current_yrs_in_comp)


        salary_1 = df1['MonthlyIncome']
        salary_1_list = [['Income']]
        for i in salary_1:
            salary_1_list.append([i])
        salary_0 = df2['MonthlyIncome']
        salary_0_list = [['Income']]
        for i in salary_0:
            salary_0_list.append([i])


        salary_1_list_json = json.dumps(salary_1_list)
        salary_0_list_json = json.dumps(salary_0_list)

    return render(request, 'blog/stats2.html',{"a" : a_json, "List_env_0_json" : List_env_0_json, "List_env_1_json" : List_env_1_json, "List_yrs_0_json" : List_yrs_0_json, "List_yrs_1_json" : List_yrs_1_json, "current_yrs_in_comp_json" : current_yrs_in_comp_json, "salary_1_list_json" : salary_1_list_json,"salary_0_list_json" : salary_0_list_json, } )  


        