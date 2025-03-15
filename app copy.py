import os
import pickle
import sqlite3

import nltk
import numpy as np
from flask import (Flask, jsonify, redirect, render_template, request, session,
                   url_for)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__, static_folder='static')

app.secret_key = os.urandom(24)  # Secret key for session management

# Load pre-trained models
with open('diabetes-prediction-model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Database setup
DB_NAME = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        username TEXT,
                        password TEXT)''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id INTEGER,
                        input_data TEXT,
                        result TEXT)''')
    conn.commit()
    conn.close()

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Sample FAQ knowledge base
FAQS = {
    "hi": "Hello! How can I assist you with your diet today?",
    "foods good for diabetes": "Foods high in fiber, such as vegetables, fruits, and whole grains, are great choices. Lean proteins like chicken and fish are also beneficial.",
    "can I eat fruits": "Yes, you can eat fruits, but prefer low-glycemic index fruits like berries, apples, and pears.",
    "what should I avoid": "You should avoid sugary drinks, refined carbs, and fried foods. Opt for whole foods instead.",
    "snacks": "Healthy snacks include nuts, seeds, Greek yogurt, and boiled eggs.",
    "dairy products": "Yes, but choose low-fat or non-fat dairy options like Greek yogurt or skim milk.",
    "thank you": "You're welcome! Feel free to ask more questions anytime.",
    "benefits of fiber": "Fiber helps in digestion, stabilizes blood sugar levels, and can reduce the risk of heart disease. It's found in whole grains, legumes, fruits, and vegetables.",
    "exercise for diabetes": "Regular exercise, such as walking, swimming, and strength training, can help manage blood sugar levels and improve insulin sensitivity.",
    "water intake": "Drinking enough water is essential for overall health, including regulating blood sugar levels. Aim for at least 8 cups a day, but individual needs may vary.",
    "meal planning": "Meal planning can help maintain blood sugar control. Focus on a balanced diet with lean proteins, whole grains, vegetables, and healthy fats.",
    "low-carb recipes": "Low-carb recipes like grilled chicken with roasted vegetables or a salad with leafy greens, avocado, and olive oil are great choices for managing diabetes.",
    "grocery shopping tips": "When shopping, focus on fresh vegetables, fruits with a low glycemic index, lean proteins, and whole grains. Avoid processed foods and sugary snacks.",
    "heart-healthy foods": "Heart-healthy foods include fatty fish like salmon, nuts, seeds, whole grains, and foods rich in antioxidants such as berries and leafy greens.",
    "managing stress": "Managing stress is key to overall health. Try deep breathing exercises, yoga, or meditation to help reduce stress levels.",
    "sleep recommendations": "Getting enough sleep (7-9 hours per night) is crucial for overall health. Aim for a consistent sleep schedule and a restful environment.",
    "low-sodium diet": "A low-sodium diet can help reduce blood pressure and promote heart health. Focus on fresh foods and avoid canned or processed foods with added salt.",
    "calcium-rich foods": "Calcium-rich foods include dairy products like yogurt and milk, as well as leafy greens like kale, and fortified non-dairy alternatives like almond milk.",
    "diabetes and alcohol": "If you drink alcohol, do so in moderation. Monitor your blood sugar levels closely and avoid drinking on an empty stomach.",
    "protein sources": "Good sources of protein include lean meats like chicken, turkey, fish, beans, lentils, tofu, and eggs.",
    "what are carbs": "Carbohydrates are a type of nutrient found in foods like grains, fruits, and vegetables. They provide energy, but it's important to choose complex carbs over refined ones.",
    "low-glycemic index foods": "Low-glycemic index foods include sweet potatoes, quinoa, legumes, berries, and apples. They have a slower effect on blood sugar levels.",
    "benefits of walking": "Walking is a great way to improve cardiovascular health, manage blood sugar levels, and boost mood. Aim for at least 30 minutes a day.",
    "vegetarian diet for diabetes": "A vegetarian diet can be great for diabetes management if planned carefully. Include plant-based proteins like beans, lentils, tofu, and plenty of vegetables and whole grains.",
    "foods that help lower cholesterol": "Foods that help lower cholesterol include oats, barley, beans, fatty fish, nuts, and foods rich in soluble fiber.",
    "what is intermittent fasting": "Intermittent fasting is a dietary approach where you cycle between periods of eating and fasting. It may help with weight management and blood sugar control.",
    "weight management tips": "For effective weight management, focus on balanced meals, portion control, regular physical activity, and staying hydrated.",
    "food portion control": "Portion control can help prevent overeating. Use smaller plates, pay attention to serving sizes, and avoid eating in front of the TV.",
    "benefits of nuts": "Nuts are rich in healthy fats, protein, and fiber. They can help improve heart health, manage blood sugar, and provide lasting energy.",
    "foods that boost metabolism": "Foods like green tea, chili peppers, and high-protein foods such as eggs and lean meats can help boost metabolism.",
    "caffeine and diabetes": "Moderate caffeine consumption is generally fine, but monitor your blood sugar levels, as caffeine can affect insulin sensitivity in some people.",
    "healthy fats": "Healthy fats are found in foods like avocados, olive oil, nuts, seeds, and fatty fish like salmon. They are beneficial for heart health.",
    "nutrient-dense foods": "Nutrient-dense foods include leafy greens, berries, lean meats, legumes, nuts, seeds, and whole grains. They provide vitamins, minerals, and fiber without excess calories.",
    "best drinks for diabetes": "Water, herbal tea, and black coffee (without sugar) are the best options. Avoid sugary drinks and sodas.",
    "best cooking oils": "Healthy cooking oils include olive oil, avocado oil, and canola oil. Avoid trans fats and hydrogenated oils.",
    "can I eat rice": "Yes, but opt for brown rice, quinoa, or cauliflower rice instead of white rice to avoid blood sugar spikes.",
    "are artificial sweeteners safe": "Artificial sweeteners like stevia and monk fruit are generally safe in moderation, but some people may prefer natural options.",
    "high-protein snacks": "High-protein snacks include boiled eggs, cottage cheese, Greek yogurt, nuts, and hummus with veggies.",
    "how to curb sugar cravings": "Eat more protein and fiber, stay hydrated, and choose natural sweeteners like fruit to help curb sugar cravings.",
    "benefits of omega-3": "Omega-3 fatty acids help reduce inflammation, support heart health, and improve brain function. Good sources include fatty fish, flaxseeds, and walnuts.",
    "importance of hydration": "Staying hydrated helps maintain energy levels, improves digestion, and supports blood sugar regulation.",
    "how often should I eat": "Eating smaller, balanced meals every 3-4 hours can help maintain stable blood sugar levels.",
    "fast food options for diabetes": "If eating fast food, choose grilled proteins, salads, and avoid high-carb or fried options.",
    "should I count calories": "Counting calories can help with weight management, but focusing on nutrient-dense foods is more important than just the numbers.",
    "best way to cook vegetables": "Steaming, roasting, or sautéing with healthy oils is the best way to preserve nutrients in vegetables.",
    "low-calorie desserts": "Try Greek yogurt with berries, dark chocolate, or chia seed pudding as healthier dessert options.",
    "why is fiber important": "Fiber helps with digestion, blood sugar control, and heart health. Found in whole grains, vegetables, and legumes.",
    "how does sleep affect diabetes": "Poor sleep can increase insulin resistance and lead to higher blood sugar levels.",
    "best exercises for weight loss": "Cardio exercises like walking, running, and cycling, combined with strength training, are great for weight loss.",
    "is honey better than sugar": "Honey is natural but still affects blood sugar levels. Use it in moderation and monitor its impact.",
    "can I eat pasta": "Yes, but choose whole-grain or legume-based pasta and watch portion sizes.",
    "benefits of intermittent fasting": "It may help with weight loss, insulin sensitivity, and overall metabolic health, but consult your doctor before starting.",
    "how to prevent diabetes complications": "Manage blood sugar, eat a healthy diet, exercise regularly, and keep up with medical check-ups.",
    "healthy meal prep ideas": "Batch-cook lean proteins, roast vegetables, and prepare salads in advance for quick and nutritious meals.",
    "how to lower blood pressure naturally": "Reduce salt intake, exercise regularly, eat potassium-rich foods, and manage stress.",
    "is dark chocolate good for diabetes": "Yes, in moderation. Choose dark chocolate with at least 70% cocoa and no added sugar.",
    "how to eat out with diabetes": "Opt for grilled proteins, salads, and whole grains. Avoid sugary drinks and refined carbs.",
    "importance of probiotics": "Probiotics support gut health and digestion. Found in yogurt, kefir, and fermented foods like kimchi.",
    "healthy smoothie ideas": "Blend spinach, banana, almond butter, and chia seeds for a nutritious smoothie.",
    "how to read food labels": "Check for added sugars, refined carbs, unhealthy fats, and high sodium content on labels.",
    "best breakfast for stable blood sugar": "Oatmeal with nuts, Greek yogurt with seeds, or eggs with avocado are great options.",
    "is fasting good for diabetes": "Intermittent fasting may help with blood sugar control, but consult your doctor before trying it.",
    "can I drink milk": "Yes, but choose low-fat or unsweetened non-dairy alternatives like almond or soy milk.",
    "best ways to reduce stress": "Try meditation, yoga, deep breathing, and getting enough sleep to manage stress effectively.",
    "importance of magnesium": "Magnesium helps with muscle function, nerve health, and blood sugar regulation. Found in nuts, seeds, and leafy greens.",
    "can I drink alcohol": "If you drink, do so in moderation and avoid sugary cocktails. Monitor your blood sugar levels closely.",
    "is popcorn a healthy snack": "Yes, if air-popped and unsweetened. Avoid buttered or caramelized varieties.",
    "how to manage cravings": "Drink water, eat protein and fiber-rich foods, and avoid skipping meals.",
    "best fruits for digestion": "Bananas, apples, berries, and papayas are great for digestion due to their fiber content.",
    "how to increase energy levels": "Eat balanced meals, stay hydrated, get enough sleep, and exercise regularly.",
    "what are whole foods": "Whole foods are minimally processed and include fruits, vegetables, whole grains, lean proteins, and healthy fats.",
    "what is mindful eating": "Mindful eating involves paying attention to your food, eating slowly, and recognizing hunger and fullness cues.",
    "best bedtime snacks": "Greek yogurt, nuts, or a small handful of berries can be good bedtime snacks.",
    "does stress affect blood sugar": "Yes, stress can increase cortisol levels, leading to higher blood sugar levels.",
    "why should I avoid processed foods": "Processed foods often contain added sugars, unhealthy fats, and preservatives that can negatively impact health.",
    "how to build a balanced plate": "Fill half your plate with vegetables, one-quarter with lean protein, and one-quarter with whole grains or healthy carbs.",
    "are legumes good for diabetes": "Yes, beans, lentils, and chickpeas are excellent sources of fiber and protein with a low glycemic impact.",
    "best bedtime routine for health": "Avoid screens before bed, have a relaxing routine, and maintain a consistent sleep schedule.",
    "does spicy food affect blood sugar": "Spicy food generally does not affect blood sugar, but some people may experience digestive discomfort.",
    "what is the best type of bread": "Whole grain, sprouted grain, or sourdough bread are better options than white bread.",
    "best way to store fresh produce": "Keep leafy greens in the fridge, store root vegetables in a cool place, and keep fruits like bananas at room temperature.",
    "should I take supplements": "A balanced diet is best, but some people may benefit from vitamin D, magnesium, or omega-3 supplements.",
    "what is HbA1c": "HbA1c (Hemoglobin A1c) is a blood test that measures your average blood sugar levels over the past 2-3 months. It helps monitor diabetes management.",
    "normal HbA1c levels": "For most people, a normal HbA1c level is below 5.7%. Prediabetes is between 5.7% and 6.4%, and diabetes is diagnosed at 6.5% or higher.",
    "how to lower HbA1c": "Lower your HbA1c by maintaining a balanced diet, exercising regularly, reducing sugar intake, and taking medications as prescribed by your doctor.",
    "how often should I check HbA1c": "People with diabetes should check their HbA1c levels at least every 3-6 months to monitor blood sugar control.",
    "why is HbA1c important": "HbA1c is important because it gives an overall picture of long-term blood sugar control, helping to prevent diabetes complications.",
    "what affects HbA1c levels": "Diet, exercise, stress, medications, and overall blood sugar management can all impact HbA1c levels.",
    "can HbA1c be inaccurate": "Yes, conditions like anemia, kidney disease, and certain hemoglobin disorders can affect the accuracy of HbA1c readings.",
    "HbA1c vs blood sugar test": "HbA1c measures average blood sugar over months, while a regular blood glucose test shows your current blood sugar level at a specific time.",
    "target HbA1c for diabetics": "The target HbA1c for most people with diabetes is below 7%, but your doctor may set a different goal based on your health condition.",
    "does fasting affect HbA1c": "No, fasting does not affect HbA1c results because it measures long-term blood sugar levels rather than daily fluctuations.",
    "can HbA1c be reversed": "Yes, by making lifestyle changes, maintaining a healthy diet, exercising, and managing stress, you can lower your HbA1c over time.",
    "does high HbA1c mean diabetes": "A high HbA1c suggests poor blood sugar control and can indicate diabetes, but a doctor will confirm with additional tests.",
    # Breakfast responses (Vegan, Vegetarian, Non-Vegetarian)
    "vegan breakfast ideas": "For a vegan breakfast, try chia pudding with almond milk and berries, a smoothie with spinach, banana, and almond butter, or avocado toast with tomato.",
    "vegetarian breakfast ideas": "Vegetarian breakfast options include scrambled eggs with spinach, Greek yogurt with berries and seeds, or whole grain toast with avocado and poached eggs.",
    "non-vegetarian breakfast ideas": "Non-vegetarian breakfast ideas include scrambled eggs with turkey bacon, a breakfast burrito with chicken and veggies, or smoked salmon on whole-grain toast.",
    
    # Lunch responses (Vegan, Vegetarian, Non-Vegetarian)
    "vegan lunch ideas": "For lunch, try a quinoa salad with chickpeas, roasted sweet potatoes with avocado, or a lentil and vegetable stew.",
    "vegetarian lunch ideas": "Vegetarian lunch options could include a veggie-packed quinoa bowl, a Greek salad with feta cheese, or a vegetable stir-fry with tofu.",
    "non-vegetarian lunch ideas": "Non-vegetarian lunch ideas include grilled chicken with roasted vegetables, a turkey and avocado wrap, or grilled salmon with quinoa and steamed broccoli.",
    
    # Dinner responses (Vegan, Vegetarian, Non-Vegetarian)
    "vegan dinner ideas": "Vegan dinner options could include a vegetable stir-fry with tofu, lentil curry, or a chickpea and spinach salad.",
    "vegetarian dinner ideas": "Vegetarian dinner options include a vegetable pasta with marinara sauce, a tofu stir-fry with vegetables, or a chickpea and vegetable stew.",
    "non-vegetarian dinner ideas": "Non-vegetarian dinner ideas include grilled chicken with roasted vegetables, baked salmon with a side of greens, or a turkey meatball soup with zucchini noodles."
}
images = {
    "walking": "/static/images/walking.jpg",
    "strength_training": "/static/images/weight.jpg",
    "yoga": "/static/images/yoga.jpg",
    "swimming": "/static/images/swimming.jpg",
    "cycling": "/static/images/cycling.jpg",
    "dancing":"/static/images/dancing.jpg",
    "jumba":"/static/images/jumba.jpg",
    "jogging":"/static/images/jogging.jpg",
    "hiking":"/static/images/hiking.jpg",
    "rowing":"/static/images/rowing.jpg",
    "skipping":"/static/images/skipping.jpg",
    "jumping_jacks":"/static/images/jumping jacks.jpg",
    "rope":"/static/images/rope.jpg",
    "aerobics":"/static/images/aerobics.jpg",
    "kai chi":"/static/images/kai chi.jpg"
}


# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(words)

# Prepare the FAQ data
faq_keys = list(FAQS.keys())
preprocessed_faq_keys = [preprocess_text(key) for key in faq_keys]

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer().fit(preprocessed_faq_keys)
faq_vectors = vectorizer.transform(preprocessed_faq_keys)

# Function to find the best matching FAQ
def find_best_match(question):
    question = preprocess_text(question)
    question_vector = vectorizer.transform([question])
    similarity_scores = cosine_similarity(question_vector, faq_vectors)
    best_match_index = similarity_scores.argmax()
    return faq_keys[best_match_index], FAQS[faq_keys[best_match_index]]

# Function to handle chatbot response
def chatbot_response(question):
    question, answer = find_best_match(question)
    satisfactory = "not satisfied" not in question.lower()
    return answer, satisfactory

# Home route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/exercise')
def exercise():
    return render_template('exercise.html', images=images)

# Chatbot route
@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'answer': 'Please ask a question!', 'satisfactory': False})
        answer, satisfactory = chatbot_response(question)
        return jsonify({'answer': answer, 'satisfactory': satisfactory})
    return render_template('chatbot.html')  # Render chatbot UI for GET request



# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE username = ? AND password = ?', (username, password))
        user = cursor.fetchone()
        conn.close()
        if user:
            session['user_id'] = user[0]
            session['username'] = username
            return redirect(url_for('home'))
        else:
            return 'Invalid credentials'
    return render_template('login.html')

# Register route
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
        conn.commit()
        conn.close()
        return redirect(url_for('login'))
    return render_template('register.html')

# Diabetes Prediction Route
@app.route('/test', methods=['GET', 'POST'])
def test():
    if request.method == 'POST':
        try:
            age = int(request.form['Age'])
            bmi = float(request.form['BMI'])
            insulin = int(request.form['Insulin'])
            glucose = int(request.form['Glucose'])
            family_history = request.form['FamilyHistory'].strip().lower()

            if family_history not in label_encoder.classes_:
                return f"<h1>Error: Unrecognized family history value: {family_history}</h1>"

            family_history_numeric = label_encoder.transform([family_history])[0]
            input_data = np.array([[age, bmi, insulin, glucose, family_history_numeric]])
            prediction = model.predict(input_data)[0]

            result = ['No Diabetes', 'Type 1 Diabetes', 'Type 2 Diabetes'][prediction] if prediction in [0, 1, 2] else 'Unknown Result'

            if 'user_id' in session:
                conn = sqlite3.connect(DB_NAME)
                cursor = conn.cursor()
                cursor.execute('INSERT INTO history (user_id, input_data, result) VALUES (?, ?, ?)', 
                               (session['user_id'], str(input_data.tolist()), result))
                conn.commit()
                conn.close()

            return render_template('result.html', result=result)

        except Exception as e:
            return f'<h1>Error: {str(e)}</h1>'

    return render_template('test.html')

@app.route('/account')
def account():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Fetch history data from database
    with sqlite3.connect(DB_NAME) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT input_data, result FROM history WHERE user_id = ?', (session['user_id'],))
        history = cursor.fetchall()

    return render_template('account.html', username=session['username'], history=history)
# Logout route
@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# Initialize database and run app
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
