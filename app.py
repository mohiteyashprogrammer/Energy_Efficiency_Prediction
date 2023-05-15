from flask import Flask,request,jsonify,render_template
from src.pipline.prediction_pipline import PredictPipline,CustomData


application = Flask(__name__)
app =application

@app.route("/",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
        relative_compactness = float(request.form.get("relative_compactness")),
        wall_area = float(request.form.get("wall_area")),
        overall_height = float(request.form.get("overall_height")), 
        orientation = int(request.form.get("orientation")),
        glazing_area = float(request.form.get("glazing_area")),
        glazing_area_distribution = int(request.form.get("glazing_area_distribution"))
        )

    final_data = data.get_data_as_data_frame()
    predict_pipline = PredictPipline()
    pred = predict_pipline.predict(final_data)

    result = pred


    return render_template("form.html",final_result = "Your Heating Load And, Cooling Load is: {}".format(result))
            
    


if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)
