from flask import Flask, request, redirect
from flask import render_template
import pickle

# creates a Flask application, named app
app = Flask(__name__)

# a route where we will display a welcome message via an HTML template
@app.route('/', methods=['GET','POST'])
def server():
    return render_template('index.html')

@app.route('/el/', methods=['GET','POST'])
def el_server():
    return render_template('el/index.html')




@app.route('/el/gender/', methods=['GET', 'POST'])
def el_gender_server():
    data = request.data or request.form
    if request.method == 'POST':
        with open(f'el_gender.pkl', 'wb') as f:
            pickle.dump(data.to_dict(), f)
        return redirect('/el/')
    return render_template('el/Gender.html')

@app.route('/el/person/', methods=['GET', 'POST'])
def el_person_server():
    data = request.data or request.form
    if request.method == 'POST':
        with open(f'el_person.pkl', 'wb') as f:
            pickle.dump(data.to_dict(), f)
        return redirect('/el/')
    return render_template('el/Person.html')

@app.route('/el/number/', methods=['GET', 'POST'])
def el_numer_server():
    data = request.data or request.form
    if request.method == 'POST':
        with open(f'el_number.pkl', 'wb') as f:
            pickle.dump(data.to_dict(), f)
        return redirect('/el/')
    return render_template('el/Number.html')

@app.route('/el/tense/', methods=['GET', 'POST'])
def el_tense_server():
    data = request.data or request.form
    if request.method == 'POST':
        with open(f'el_tense.pkl', 'wb') as f:
            pickle.dump(data.to_dict(), f)
        return redirect('/el/')
    return render_template('el/Tense.html')

@app.route('/el/case/', methods=['GET', 'POST'])
def el_case_server():
    data = request.data or request.form
    if request.method == 'POST':
        with open(f'el_case.pkl', 'wb') as f:
            pickle.dump(data.to_dict(), f)
        return redirect('/el/')
    return render_template('el/Case.html')


@app.route('/el/mood/', methods=['GET', 'POST'])
def el_mood_server():
    data = request.data or request.form
    if request.method == 'POST':
        with open(f'el_mood.pkl', 'wb') as f:
            pickle.dump(data.to_dict(), f)
        return redirect('/el/')
    return render_template('el/Mood.html')


# run the application
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
