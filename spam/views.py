from django.shortcuts import render
from django.http import HttpResponse
import os
import joblib



vectorizer = joblib.load(os.path.join(os.path.dirname(__file__), "vectorizer.joblib"))

nb_classifier = joblib.load(os.path.join(os.path.dirname(__file__), "nb_classifier.joblib"))

rnn_model = joblib.load(os.path.join(os.path.dirname(__file__), "rnnspam.pkl"))



# mymdl = joblib.load(os.path.dirname(__file__) + "\\nbspam.pkl")

# mymd2 = joblib.load(os.path.dirname(__file__) + "\\rnnspam.pkl")


# Create your views here.
def index(request):
    return render(request, "index.html")

def analysis(request):
    return render(request, "analysis.html")

def about(request):
    return render(request, "about.html")


def checkSpam(request):
    param = {}
    if(request.method == "POST"):
      
        alg = request.POST.get("model")

        rawdata = request.POST.get("rawdata")

        input_data = [rawdata]

        input_data_vectorized = vectorizer.transform(input_data)
        
        if(alg == "alg-1"):
            # vectorizer = mymdl.vectorizer
            # nb_classifier = mymdl.classifier
            # input_data_vectorized = vectorizer.transform(input_data)

            numeric_label = nb_classifier.predict(input_data_vectorized)[0]
            predicted_class = "spam" if numeric_label == 1 else "ham"
            param = {"answer": predicted_class}
        # elif(alg == "alg-2"):
        #     finalans = mymd2.predict([rawdata])
        #     param = {"answer" : predicted_class}

          

        return render(request , 'output.html' , param)  
        
    else:
        return render(request, "index.html")
