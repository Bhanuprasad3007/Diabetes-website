@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question', '').strip()
        if not question:
            return jsonify({'answer': 'Please ask a question!', 'satisfactory': False})
        answer, satisfactory = chatbot_response(question)
        return jsonify({'answer': answer, 'satisfactory': satisfactory})
    return render_template('chatbot.html') 