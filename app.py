import numpy as np
from flask import Flask, request, jsonify, render_template
app = Flask(__name__)

# page index principale qui appelle index.html. En fait, pour utiliser le modele on fera 127.0.0.1:5000 dans le navigateur
@app.route('/')
def home():
   return render_template('index.html')
#La methode predict est appelee dans  action="{{ url_for('predict')}}" du formulaire html
@app.route('/predict',methods=['POST'])
def predict():
    import joblib
    model = joblib.load('ModelPredireAgeMariage.ml') #J'utilise joblib.load pour charger notre modele sauvegarde avec joblib.dump()
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    indexDatenaisance=[1,2,3,4,5,6]
    dateNaissance=np.delete(final_features,indexDatenaisance)
    dateNaissance=dateNaissance[0]
    #naissance=np.array([dateNaissance]).reshape(1,1)
    #Je vais enlever la valeur de annee de naissance car le model n'attend que 6 features
    #Cette annee va m'aider a predire l'annee de mariage en fonction de l'age predit.
    indexAnnee=0
    features_model=np.delete(final_features,indexAnnee)
    features_model=np.array([features_model]).reshape(1,6) #Ici nous redimensionnons notre array pour qu'il redevienne de 1 ligne et 6 colonnes
    prediction = model.predict(features_model) #J'appelle notre modele qui a ete charge et je predit avec des features saisis
    AgePredit = round(prediction[0], 2) #On arrondi la valeur de sortie a 2 deimaux pres
    import math
    x = AgePredit
    out=math.modf(x)
    outRecupererPartieEntierre=out[1] #En fait, l'execution de la en haut retourne (partie decimal, partie entiere) et partie entiere a l'indice 1
    output=dateNaissance+outRecupererPartieEntierre
    
    #prediction_text est appelee dans la page html pour afficher les donnees renderisees 
    #On devra mettre index.html dans le dossier template pour que le systeme le reconnaisse
    return render_template('index.html', prediction_text='Vous devriez vous mariver en  {}'.format(output))
  

if __name__ == "__main__":
    app.run(debug=True)