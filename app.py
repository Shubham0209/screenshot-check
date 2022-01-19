import os
from flask import Flask, request, render_template
#from original_code import return_positions_confident
from resize_image import caclulate_similarity

app = Flask(__name__)

PEOPLE_FOLDER = os.path.join('static')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    urls = [str(x) for x in request.form.values()]
    print(urls)
    
    #prediction = return_positions_confident(urls)
    prediction = caclulate_similarity(urls)
    full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'Final.jpg')


    #return render_template('index.html', user_image = full_filename,prediction_text='Template image present in main image with {} % probability'.format(prediction))
    return render_template('index.html', user_image = full_filename,prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)