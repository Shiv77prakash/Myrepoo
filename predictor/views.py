from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from django.core.files.storage import FileSystemStorage

def home(request):
    prediction = None
    hours = None
    attendence= None
    

    if request.method == "POST" and request.FILES.get("csv_file"):
        # Save uploaded file
        csv_file = request.FILES["csv_file"]
        fs = FileSystemStorage()
        filename = fs.save(csv_file.name, csv_file)
        file_path = fs.path(filename)

        # Load dataset
        df = pd.read_csv(file_path)

        # Debug: print actual column names
        print("CSV Columns:", df.columns)

        
        # Use first column as feature (X), second column as target (y)
        X=df.iloc[:,:-1]
        y = df.iloc[:,-1]


        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Save model
        joblib.dump(model, "predictor/student_model.pkl")

        # If user entered hours, predict
        if request.POST.get("hours"):
            hours = float(request.POST.get("hours"))
            attendence =float(request.POST.get("attendence"))
            features= np.array([[hours,attendence]])
            prediction = model.predict(features)[0]

    return render(request, "predictor/home.html", {"prediction": prediction, "hours": hours,"attendence":attendence,"features": features})