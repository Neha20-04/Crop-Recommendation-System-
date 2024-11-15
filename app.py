from tkinter import scrolledtext
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from flask import *
from mlxtend.classifier import StackingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, Flatten
import numpy as np

import mysql.connector
db=mysql.connector.connect(user="root",password="",port='3306',database='hate_speech')
cur=db.cursor()

app=Flask(__name__)
app.secret_key="CBJcb786874wrf78chdchsdcv"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/userhome')
def userhome():
    return render_template('userhome.html')

@app.route('/crop')
def crop():
    return render_template('crop.html')

@app.route('/rain')
def rain():
    return render_template('rain.html')

@app.route('/login',methods=['POST','GET'])
def login():
    if request.method=='POST':
        useremail=request.form['useremail']
        session['useremail']=useremail
        userpassword=request.form['userpassword']
        sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
        cur.execute(sql)
        data=cur.fetchall()
        db.commit()
        if data ==[]:
            msg="user Credentials Are not valid"
            return render_template("login.html",name=msg)
        else:
            return render_template("userhome.html",myname=data[0][1])
    return render_template('login.html')
@app.route('/registration',methods=["POST","GET"])
def registration():
    if request.method=='POST':
        username=request.form['username']
        useremail = request.form['useremail']
        userpassword = request.form['userpassword']
        conpassword = request.form['conpassword']
        Age = request.form['Age']
        
        contact = request.form['contact']
        if userpassword == conpassword:
            sql="select * from user where Email='%s' and Password='%s'"%(useremail,userpassword)
            cur.execute(sql)
            data=cur.fetchall()
            db.commit()
            print(data)
            if data==[]:
                
                sql = "insert into user(Name,Email,Password,Age,Mob)values(%s,%s,%s,%s,%s)"
                val=(username,useremail,userpassword,Age,contact)
                cur.execute(sql,val)
                db.commit()
                flash("Registered successfully","success")
                return render_template("login.html")
            else:
                flash("Details are invalid","warning")
                return render_template("registration.html")
        else:
            flash("Password doesn't match", "warning")
            return render_template("registration.html")
    return render_template('registration.html')

@app.route('/view')
def view():
    dataset = pd.read_csv('Crop_recommendation.csv')
    print(dataset)
    print(dataset.head(2))
    print(dataset.columns)
    return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())

@app.route('/preprocess', methods=['POST', 'GET'])
def preprocess():
    global x, y, x_train, x_test, y_train, y_test,  hvectorizer,df
    if request.method == "POST":
        size = int(request.form['split'])
        size = size / 100
        from sklearn.preprocessing import LabelEncoder
        le=LabelEncoder()
        df = pd.read_csv('Crop_recommendation.csv')
        df.head()
        df['label'] = le.fit_transform(df['label'])

        df.head()
        df.columns
        
       # Assigning the value of x and y 
        x = df.drop(['label'], axis=1)
        y = df['label']

        x_train, x_test, y_train, y_test = train_test_split(x,y, stratify=y, test_size=size, random_state=42)


        # describes info about train and test set
        print("Number transactions X_train dataset: ", x_train.shape)
        print("Number transactions y_train dataset: ", y_train.shape)
        print("Number transactions X_test dataset: ", x_test.shape)
        print("Number transactions y_test dataset: ", y_test.shape)

    
        print(x_train,x_test)
        print(y_train)
        print(y_test)

        return render_template('preprocess.html', msg='Data Preprocessed and It Splits Successfully')
    return render_template('preprocess.html')

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
       
        print('ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc')
        s = int(request.form['algo'])
        global x, y, x_train, x_test, y_train, y_test
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            print('aaaaaaaaaaaaaaaaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb')
            dt = DecisionTreeClassifier()
            dt.fit(x_train,y_train)
            y_pred = dt.predict(x_test)
            ac_dt = accuracy_score(y_test, y_pred)
            ac_dt = ac_dt * 100
            print('aaaaaaaaaaaaaaaaaaaaaaaaa')
            msg = 'The accuracy obtained by Decision Tree Classifier is  ' + str(ac_dt) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 2:
            classifier = RandomForestClassifier()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Random Forest Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 3:
            from sklearn.ensemble import AdaBoostClassifier
            classifier = AdaBoostClassifier()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by AdaBoost Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 4:
            from xgboost import XGBClassifier
            classifier = XGBClassifier()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by XGBoost Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 5:
            from sklearn.svm import SVC
            classifier = SVC()
            classifier.fit(x_train, y_train)
            y_pred  =  classifier.predict(x_test)            
            
            ac_nb = accuracy_score(y_test, y_pred)
            ac_nb = ac_nb * 100
            msg = 'The accuracy obtained by Support Vector Classifier is ' + str(ac_nb) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 6:
            from mlxtend.classifier import StackingClassifier
            model1 =RandomForestClassifier()
            model2 = DecisionTreeClassifier()

            gnb = RandomForestClassifier()
            clf_stack = StackingClassifier(classifiers=[model1, model2], meta_classifier=gnb, use_probas=True,
                                                    use_features_in_secondary=True)
            model_stack = clf_stack.fit(x_train, y_train)
            pred_stack = model_stack.predict(x_test)
            acc_stack = accuracy_score(y_test, pred_stack)
            acc_stack
            msg = 'The accuracy obtained by Hybrid Model Classifier is ' + str(acc_stack) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 7:  
        
            model = Sequential()
            model.add(SimpleRNN(128, activation='relu', input_shape=(x_train.shape[1], 1), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(SimpleRNN(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(len(np.unique(y_train)), activation='softmax'))

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Reshape the data for RNN (samples, time steps, features)
            x_train_rnn = np.expand_dims(x_train, axis=-1)
            x_test_rnn = np.expand_dims(x_test, axis=-1)

            model.fit(x_train_rnn, y_train, epochs=50, batch_size=32, verbose=0)
            loss, ac_rnn = model.evaluate(x_test_rnn, y_test, verbose=0)
            ac_rnn = ac_rnn * 100
            msg = 'The accuracy obtained by Recurrent Neural Network is ' + str(ac_rnn) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 8:  
            model = Sequential()
            model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(len(np.unique(y_train)), activation='softmax'))

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
            loss, ac_nn = model.evaluate(x_test, y_test, verbose=0)
            ac_nn = ac_nn * 100
            msg = 'The accuracy obtained by Convolutional Neural Network is ' + str(ac_nn) + str('%')
            return render_template('model.html', msg=msg)
        elif s == 9:  
            model = Sequential()
            model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
            model.add(Dropout(0.2))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(len(np.unique(y_train)), activation='softmax'))

            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            model.fit(x_train, y_train, epochs=50, batch_size=32, verbose=0)
            loss, ac_ann = model.evaluate(x_test, y_test, verbose=0)
            ac_ann = ac_ann * 100
            msg = 'The accuracy obtained by Artificial Neural Network is ' + str(ac_ann) + str('%')
            return render_template('model.html', msg=msg)
        
    return render_template('model.html')



@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    
    if request.method == "POST":
        f1 = float(request.form['N'])
        f2 = float(request.form['P'])
        f3 = float(request.form['K'])
        f4 = float(request.form['temperature'])
        f5 = float(request.form['humidity'])
        f6 = float(request.form['ph'])
        f7 = float(request.form['rainfall'])

        # Prepare input features
        input_features = np.array([[f1, f2, f3, f4, f5, f6, f7]])
        
        # Reshape input features for CNN
        input_features = input_features.reshape((input_features.shape[0], input_features.shape[1], 1))
        global x, y, x_train, x_test, y_train, y_test
        # Reshape training data
        x_train_cnn = x_train.values.reshape((x_train.shape[0], x_train.shape[1], 1))

        # Define the CNN model
        model = Sequential()
        model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(x_train_cnn.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(Conv1D(32, kernel_size=2, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(len(np.unique(y_train)), activation='softmax'))  # Output layer

        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(x_train_cnn, y_train, epochs=50, batch_size=32, verbose=0)

        # Make prediction
        prediction = model.predict(input_features)
        result = np.argmax(prediction)  # Get the class with the highest probability

        # Map the result to the crop name
        crop_dict = {
            0: 'Apple', 1: 'Banana', 2: 'Blackgram', 3: 'Chickpea', 4: 'Coconut',
            5: 'Coffee', 6: 'Cotton', 7: 'Grapes', 8: 'Jute', 9: 'Kidneybeans',
            10: 'Lentil', 11: 'Maize', 12: 'Mango', 13: 'Mothbeans', 14: 'Moongbeans',
            15: 'Muskmelon', 16: 'Orange', 17: 'Papaya', 18: 'Pigeonpeas', 19: 'Pomegranate',
            20: 'Rice', 21: 'Watermelon'
        }

        msg = f"The Recommended Crop is predicted as {crop_dict[result]}"
        return render_template('prediction.html', msg=msg)

    return render_template('prediction.html')

if __name__=='__main__':
    app.run(debug=True)