from flask import Flask, render_template,request
import pickle
import pandas as pd
import numpy as np
import warnings
import sklearn
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
from xgboost import Booster
def prediction(arrayofinputs):
    print(arrayofinputs)
    train_data=pd.read_csv("loan-train.csv")
    test_data=pd.read_csv("loan-test.csv")
    train_copy=train_data.copy()
    train_copy["Gender"].fillna(train_copy["Gender"].mode()[0],inplace=True)
    train_copy["Married"].fillna(train_copy["Married"].mode()[0],inplace=True)
    train_copy["Dependents"].fillna(train_copy["Dependents"].mode()[0],inplace=True)
    train_copy["Self_Employed"].fillna(train_copy["Self_Employed"].mode()[0],inplace=True)
    train_copy["Credit_History"].fillna(train_copy["Credit_History"].mode()[0],inplace=True)
    train_copy["Loan_Amount_Term"].fillna(train_copy["Loan_Amount_Term"].mode()[0],inplace=True)
    train_copy["LoanAmount"].fillna(train_copy["LoanAmount"].median(), inplace=True)
    train_copy["Dependents"].replace('3+',3,inplace=True)
    train_copy["Loan_Status"].replace('Y',1,inplace=True)
    train_copy["Loan_Status"].replace('N',0,inplace=True)
    test_copy=test_data.copy()
    test_copy["Gender"].fillna(test_copy["Gender"].mode()[0],inplace=True)
    test_copy["Married"].fillna(test_copy["Married"].mode()[0],inplace=True)
    test_copy["Dependents"].fillna(test_copy["Dependents"].mode()[0],inplace=True)
    test_copy["Self_Employed"].fillna(test_copy["Self_Employed"].mode()[0],inplace=True)
    test_copy["Credit_History"].fillna(test_copy["Credit_History"].mode()[0],inplace=True)
    test_copy["Loan_Amount_Term"].fillna(test_copy["Loan_Amount_Term"].mode()[0],inplace=True)
    test_copy["LoanAmount"].fillna(test_copy["LoanAmount"].median(), inplace=True)
    test_copy["Dependents"].replace('3+',3,inplace=True)
    test_copy=test_copy.drop("Loan_ID",axis=1)
    train_copy=train_copy.drop("Loan_ID",axis=1)
    x = train_copy.drop("Loan_Status",axis=1)
    y = train_copy["Loan_Status"]
    x=pd.get_dummies(x)
    train_copy1=pd.get_dummies(train_copy)
    test_copy1=pd.get_dummies(test_copy)
    xtrain,xtest,ytrain,ytest = sklearn.model_selection.train_test_split(x,y,test_size=0.2)
    kf=sklearn.model_selection.StratifiedKFold(n_splits=7,random_state=1,shuffle=True)
    XGB=XGBClassifier(random_state=1,n_estimators=20)
    XGB.fit(np.array(xtrain),np.array(ytrain))
    ans=XGB.predict(arrayofinputs)
    return ans

app = Flask(__name__)

@app.route('/')  # url binding
def loadhome():
    return render_template('Home.html')


@app.route('/form')
def dataform():
    return render_template('DATA_ENTRY.html',res = None)


@app.route('/submit', methods=['POST'])  # url binding
def user():
    print(request.form)
    Education = request.form['Education']
    ApplicantIncome = request.form['ApplicantIncome']
    Coapplicant = request.form['CoapplicantIncome']
    LoanAmount = request.form['LoanAmount']
    LoanAmountTerm = request.form['Loan_Amount_Term']
    CreditHistory = request.form['Credit_History']
    dependents = request.form['Dependents']
    property = request.form['Property_Area']
    gender = request.form['Gender']
    married=request.form['Married']
    self_employed=request.form['Self_Employed']
    if Education == 'Graduate':
        se1,se2 = 1,0
    else:
        se1,se2 = 0,1
    if dependents == '0':
        s3, s0, s1, s2 = 0, 1, 0, 0
    elif dependents == '1':
        s3, s0, s1, s2 = 0, 0, 1, 0
    elif dependents == '2':
        s3, s0, s1, s2 = 0, 0, 0, 1
    elif dependents == '3+':
        s3, s0, s1, s2 = 1, 0, 0, 0
    if property == 'Rural':
        sp1, sp2, sp3 = 1, 0, 0
    elif property == 'Semiurban':
        sp1, sp2, sp3 = 0, 1, 0
    elif property == 'Urban':
        sp1, sp2, sp3 = 0, 0, 1
    if gender == 'Female':
        sg1,sg2=1,0
    elif gender=='Male':
        sg1,sg2=0,1
    if married=='Yes':
        sm1,sm2=0,1
    else:
        sm1,sm2=1,0
    if self_employed == 'Yes':
        semp1,semp2=0,1
    else:
        semp1,semp2=1,0
    # Education, Applicant Income, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, dependents(4), property(3)

    arrayofinputs = [[float(ApplicantIncome), float(Coapplicant), float(
    LoanAmount), float(LoanAmountTerm), float(CreditHistory), float(sg1), float(sg2),float(sm1),float(sm2), float(s3), float(s0), float(s1), float(s2), float(se1), float(se2), 
    float(semp1),float(semp2),float(sp1),float(sp2),float(sp3)]]

    ans=prediction(arrayofinputs)
    if  ans == 0:
        print(ans)
        result='Rejected'
    else:
        print(ans)
        result='Approved'
    result = str(result)
    print(result)
    return render_template('DATA_ENTRY.html',res = result)

if __name__ == '__main__':
   app.run()