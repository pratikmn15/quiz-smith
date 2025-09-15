from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import json
import os
from datetime import datetime
import glob

app = Flask(__name__)
app.secret_key = 'quiz-smith-secret-key-2025' 

def load_mcq_files():
    """Load all available MCQ JSON files."""
    mcq_files = glob.glob("mcqs_*.json")
    file_info = []
    
    for file in mcq_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                file_info.append({
                    'filename': file,
                    'query': data['metadata']['query'],
                    'total_questions': data['metadata']['total_questions'],
                    'generated_at': data['metadata']['generated_at'],
                    'display_date': datetime.fromisoformat(data['metadata']['generated_at']).strftime('%Y-%m-%d %H:%M')
                })
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Sort by generation time (newest first)
    file_info.sort(key=lambda x: x['generated_at'], reverse=True)
    return file_info

def load_mcq_data(filename):
    """Load MCQ data from a specific JSON file."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return None

@app.route('/')
def index():
    """Home page showing available MCQ files."""
    mcq_files = load_mcq_files()
    return render_template('index.html', mcq_files=mcq_files)

@app.route('/quiz/<filename>')
def start_quiz(filename):
    """Start a quiz with the selected MCQ file."""
    mcq_data = load_mcq_data(filename)
    if not mcq_data:
        return redirect(url_for('index'))
    
    # Initialize session with minimal data
    session['quiz_filename'] = filename
    session['current_question'] = 0
    session['answers'] = {}
    session['score'] = 0
    session['quiz_started'] = True
    
    return redirect(url_for('question'))

@app.route('/question')
def question():
    """Display current question."""
    if 'quiz_filename' not in session:
        return redirect(url_for('index'))
    
    quiz_data = load_mcq_data(session['quiz_filename'])
    current_q = session['current_question']
    
    if current_q >= len(quiz_data['questions']):
        return redirect(url_for('results'))
    
    question_data = quiz_data['questions'][current_q]
    total_questions = len(quiz_data['questions'])
    
    return render_template('question.html', 
                         question=question_data,
                         current=current_q + 1,
                         total=total_questions,
                         quiz_title=quiz_data['metadata']['query'])

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    """Submit answer for current question."""
    if 'quiz_filename' not in session:
        return jsonify({'error': 'No active quiz'}), 400
    
    quiz_data = load_mcq_data(session['quiz_filename'])
    user_answer = request.json.get('answer')
    current_q = session['current_question']
    
    if current_q >= len(quiz_data['questions']):
        return jsonify({'error': 'Quiz completed'}), 400
    
    question_data = quiz_data['questions'][current_q]
    correct_answer = question_data['correct_answer']
    
    # Store the answer (stringify key)
    session['answers'][str(current_q)] = {
        'user_answer': user_answer,
        'correct_answer': correct_answer,
        'is_correct': user_answer == correct_answer
    }
    
    # Update score
    if user_answer == correct_answer:
        session['score'] += 1
    
    # Move to next question
    session['current_question'] += 1
    
    return jsonify({
        'correct': user_answer == correct_answer,
        'correct_answer': correct_answer,
        'explanation': f"The correct answer is {correct_answer}"
    })

@app.route('/results')
def results():
    """Display quiz results."""
    if 'quiz_filename' not in session or 'answers' not in session:
        return redirect(url_for('index'))
    
    quiz_data = load_mcq_data(session['quiz_filename'])
    answers = session['answers']
    score = session['score']
    total = len(quiz_data['questions'])
    percentage = (score / total * 100) if total > 0 else 0
    
    # Generate detailed results
    detailed_results = []
    for i, question in enumerate(quiz_data['questions']):
        if str(i) in answers:  # use str(i)
            detailed_results.append({
                'question': question,
                'user_answer': answers[str(i)]['user_answer'],
                'correct_answer': answers[str(i)]['correct_answer'],
                'is_correct': answers[str(i)]['is_correct']
            })
    
    return render_template('results.html',
                         quiz_title=quiz_data['metadata']['query'],
                         score=score,
                         total=total,
                         percentage=percentage,
                         detailed_results=detailed_results)

@app.route('/reset')
def reset_quiz():
    """Reset current quiz session."""
    session.clear()
    return redirect(url_for('index'))

@app.route('/api/quiz_progress')
def quiz_progress():
    """API endpoint to get current quiz progress."""
    if 'quiz_filename' not in session:
        return jsonify({'error': 'No active quiz'}), 400
    
    quiz_data = load_mcq_data(session['quiz_filename'])
    return jsonify({
        'current_question': session['current_question'],
        'total_questions': len(quiz_data['questions']),
        'score': session['score']
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
