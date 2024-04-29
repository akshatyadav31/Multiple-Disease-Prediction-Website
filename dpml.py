from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import *
from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
import warnings
import csv

# 2. App 
app = Flask(__name__, static_folder='static')

feature = [
    'back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine','yellowing_of_eyes',
    'acute_liver_failure','fluid_overload','swelling_of_stomach','swelled_lymph_nodes','malaise',
    'blurred_and_distorted_vision','phlegm','throat_irritation','redness_of_eyes','sinus_pressure','runny_nose',
    'congestion','chest_pain','weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region',
    'bloody_stool','irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
    'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
    'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain','hip_joint_pain',
    'muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements','loss_of_balance',
    'unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
    'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)','depression','irritability',
    'muscle_pain','altered_sensorium','red_spots_over_body','belly_pain','abnormal_menstruation','dischromic _patches',
    'watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum','rusty_sputum',
    'lack_of_concentration','visual_disturbances','receiving_blood_transfusion','receiving_unsterile_injections','coma',
    'stomach_bleeding','distention_of_abdomen','history_of_alcohol_consumption','fluid_overload','blood_in_sputum',
    'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
    'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose','yellow_crust_ooze'
]

symp = {}
for i in sorted(feature):
    j = str(i).replace("_"," ")
    symp[i] = str(i).replace("_"," ").upper()

disease = [
    'Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction','Peptic ulcer diseae','AIDS','Diabetes',
    'Gastroenteritis','Bronchial Asthma','Hypertension','Migraine','Cervical spondylosis','Paralysis (brain hemorrhage)',
    'Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A','Hepatitis B','Hepatitis C','Hepatitis D',
    'Hepatitis E','Alcoholic hepatitis','Tuberculosis','Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
    'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis','Arthritis',
    '(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis','Impetigo'
]

l2 = [0] * len(feature)

Training_Data=pd.read_csv("DISEASE_TRAIN.csv")
Testing_Data=pd.read_csv("DISEASE_TEST.csv")

for i in [Testing_Data, Training_Data]:
    i.replace(
    {
        'prognosis':{
            'Fungal infection':0,'Allergy':1,'GERD':2,
            'Chronic cholestasis':3,'Drug Reaction':4,
            'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,
            'Gastroenteritis':8,'Bronchial Asthma':9,
            'Hypertension ':10,'Migraine':11,'Cervical spondylosis':12,
            'Paralysis (brain hemorrhage)':13,'Jaundice':14,
            'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,
            'hepatitis A':19,'Hepatitis B':20,'Hepatitis C':21,
            'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,
            'Tuberculosis':25,'Common Cold':26,'Pneumonia':27,
            'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,
            'Varicose veins':30,'Hypothyroidism':31,'Hyperthyroidism':32,
            'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
            '(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,
            'Urinary tract infection':38,'Psoriasis':39,'Impetigo':40
        }
    },inplace=True
)


X_train = Training_Data[feature]
y_train = Training_Data[["prognosis"]]
X_test = Testing_Data[feature]
y_test = Testing_Data[["prognosis"]]
y_train=np.ravel(y_train)
x_train = X_train


clf0 =  svm.SVC(
    kernel='linear',
    probability=True
    )
clf1 = KNeighborsClassifier(
    n_neighbors=4
    )
clf2 = RandomForestClassifier(
    n_estimators = 100
    )
clf3 = VotingClassifier(
    estimators=[("SVM",clf0),("KNN",clf1),("RandomF",clf2)],
    voting='soft'
    )
clf3 = clf3.fit(x_train,y_train)


@app.route('/')
def index():
    return render_template('index.html',symptons= symp)

@app.route("/static/favicon.ico") 
def favicon(): 
    return send_from_directory(app.static_folder, 'favicon.ico') 

@app.route("/static/styles.css") 
def favstyle(): 
    return send_from_directory(app.static_folder, 'styles.css') 


@app.route("/static/styles1.css") 
def favstyle1(): 
    return send_from_directory(app.static_folder, 'styles1.css')
 

@app.route('/web-WEB/predict', methods=['POST'])
def predict():
    symptoms_ = [
        request.form['Symptom1'], request.form['Symptom2'], 
        request.form['Symptom3'], request.form['Symptom4'], 
        request.form['Symptom5'], request.form['Symptom6'], 
        request.form['Symptom7'], request.form['Symptom8'], 
        request.form['Symptom9'], request.form['Symptom10'], 
        request.form['Symptom11'], request.form['Symptom12'],
        request.form['Symptom13'], request.form['Symptom14'], 
        request.form['Symptom15'],request.form['Symptom16'], 
        request.form['Symptom17']
    ]
    TopNresult = int(request.form['TopN'])
    symptoms_text = [word.strip().replace(" ", "_").lower() for word in symptoms_]

    for i in range(0, len(feature)):
        for j in symptoms_text:
            if j.lower() == feature[i]:
                l2[i] = 1
    
    input_user = [l2]
    predict_proba = clf3.predict_proba(input_user)[0]
    
    top5_diseases_default = ["Drug Reaction"]*5
    
    top5_indices = np.argsort(predict_proba)[::-1][:TopNresult]
   
    top5_diseases = [disease[i] for i in top5_indices]
    if (symptoms_text==[""]*17):
        top5_diseases= top5_diseases_default
    
    # Append prediction to history CSV file
    with open('prediction.csv', 'a', newline='') as csvfile:
        fieldnames = ['Symptoms', 'Predicted_Diseases']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Combine symptoms into a string
        symptoms_str = ', '.join(symptoms_text)
        
        # Combine top 5 predicted diseases into a string
        diseases_str = ', '.join(top5_diseases)
        
        # Write the row to the CSV file
        writer.writerow({'Symptoms': symptoms_str, 'Predicted_Diseases': diseases_str})

    return render_template(
        'result.html', 
        topvalue=TopNresult, 
        top5_diseases=top5_diseases
    )
    
    
@app.route('/history')
def history():
    history_data = []
    with open('prediction.csv', 'r', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Split the row into symptoms and predicted diseases
            symptoms = row[0].split(',')
            diseases = row[1].split(',')
            
            # Remove any empty strings and leading/trailing whitespace
            symptoms = [symptom.strip() for symptom in symptoms if symptom.strip()]
            diseases = [disease.strip() for disease in diseases if disease.strip()]
            
            # Append the cleaned data to history_data
            history_data.append({'Symptoms': symptoms, 'Predicted_Diseases': diseases})

    return render_template('history.html', history_data=history_data)


@app.route('/about')
def about():
    # Your about page logic goes here
    return render_template('about.html')
if __name__ == '__main__':
    app.run(debug=True,port=57578)





# clf0 = clf0.fit(x_train,y_train)
# y_pred0 = clf0.predict(x_test)
# y_pred_0 = y_pred0#[0]


# clf1 = clf1.fit(x_train,y_train)
# y_pred1 = clf1.predict(x_test)
# y_pred_1 = y_pred1#[0]


# clf2 = clf2.fit(x_train,y_train)
# y_pred2 = clf2.predict(x_test)
# y_pred_2 = y_pred2#[0]

# acc = accuracy_score(np.ravel(y_test),clf_test)
# print(acc*100)
    
# return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', minetype='image/vnd.microsof.icon')
 # return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', minetype='image/vnd.microsof.icon')
       
    # for i in symptoms_text:
    #     print("Symptons:",i, i.__contains__(" "), len(i))

# for i in range(len(predict_proba)-1,5,-1):
    #     print(predict_proba[i])

   # print(top5_diseases_default)
    # print(top5_indices)  

# @app.route('/web-WEB/predict#', methods=['POST'])
# def home_nav():
#     return render_template('index.html',symptons= symp)
# @app.route('/#', methods=['POST'])
# def home_():
#     return render_template('index.html',symptons= symp)
