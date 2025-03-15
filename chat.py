import json
import random

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# Load predefined responses from a JSON file
responses = {
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

# Simple text preprocessing
def preprocess(text):
    return text.lower().strip()

# Get response based on user input
def get_response(user_input):
    user_input = preprocess(user_input)
    for key in responses:
        if key in user_input:
            return responses[key]
    return "I'm not sure about that. Please ask another question."

@app.route('/')
def home():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    response = get_response(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
