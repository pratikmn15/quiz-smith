# Generate MCQs using HuggingFace Inference API (modern calls)
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from query_database import search_database  # your real function

# Load environment variables
load_dotenv()
HF_LLM_API_KEY = os.getenv("HF_LLM_API_KEY")  # keep same name in .env
MODEL_ID = "Qwen/Qwen3-14B"            # same model as test.py

def setup_llm():
    """Initialize the Hugging Face InferenceClient for generation."""
    print("Setting up Hugging Face InferenceClient for question generation...")
    if not HF_LLM_API_KEY:
        print("❌ HF_LLM_API_KEY not found in environment variables.")
        print("Please add your HuggingFace API key to the .env file: HF_LLM_API_KEY=your_hf_api_key_here")
        return None
    
    # Debug: Show first few characters of API key
    print(f"🔑 Using API key: {HF_LLM_API_KEY[:10]}...")
    
    try:
        client = InferenceClient(
            provider="nebius",
            api_key=HF_LLM_API_KEY,
        )
        print("✅ InferenceClient initialized.")
        return client
    except Exception as e:
        print(f"❌ Error initializing InferenceClient: {e}")
        return None

def test_hf_llm_api():
    """Test if Hugging Face Inference API is working properly."""
    print("Testing Hugging Face Inference API connection...")
    client = setup_llm()
    if not client:
        return False
    
    try:
        test_prompt = "What is 2+2?"
        print(f"🧪 Testing with prompt: '{test_prompt}'")
        print(f"🎯 Testing model: {MODEL_ID}")
        
        # Use the exact same API call as test.py
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": test_prompt
                }
            ],
        )
        
        response_text = completion.choices[0].message.content
        print(f"📝 Extracted text: '{response_text}'")
        
        if response_text and response_text.strip():
            print("✅ Hugging Face LLM API is working!")
            print(f"   Model: {MODEL_ID}")
            print(f"   Test response length: {len(response_text)} characters")
            return True
        else:
            print("❌ Hugging Face LLM API returned empty response.")
            return False
            
    except Exception as e:
        print(f"❌ Hugging Face LLM API error: {e}")
        print(f"❌ Error type: {type(e)}")
        
        # Check if it's an authentication error
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("🔑 This looks like an authentication error. Check your API key.")
        elif "403" in str(e) or "forbidden" in str(e).lower():
            print("🚫 This looks like a permission error. Your API key might not have access to this model.")
        elif "404" in str(e) or "not found" in str(e).lower():
            print("🔍 Model not found. The model might not be available via Inference API.")
        
        print("Please check:")
        print("1. Your HF_LLM_API_KEY is valid and active")
        print("2. Your API key has access to the Inference API")
        print("3. The model Qwen/Qwen3-14B is available via Inference API")
        print("4. Your network connection")
        
        return False

def create_mcq_prompt(context, num_questions):
    """Create a concise instruction-based prompt for Qwen models."""
    return f"""Generate {num_questions} multiple choice questions from this study material.

Study Material:
{context}

For each question:
- Create exactly 4 answer choices (A, B, C, D)
- Make one choice correct and three plausible but incorrect
- Prefer conceptual/testing-of-understanding style questions

Format each question exactly like this:
Question 1: [question text]
A) [option A]
B) [option B]
C) [option C]
D) [option D]
Correct Answer: [A/B/C/D]

Generate all {num_questions} questions now:"""

def generate_mcqs_from_content(content, num_questions=5):
    """Generate MCQs from provided content using Hugging Face Inference API."""
    if not content:
        print("❌ No content provided for MCQ generation.")
        return None

    client = setup_llm()
    if not client:
        return None

    try:
        # Truncate content if too long for model context
        max_content_length = 3000  # Increased for Qwen model
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
            print(f"⚠️  Content truncated to {max_content_length} characters for model processing")

        prompt = create_mcq_prompt(content, num_questions)
        print("🤖 Generating questions with HuggingFace Inference API...")

        # Use the exact same API call structure as test.py
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
        )
        
        generated_text = completion.choices[0].message.content

        if generated_text and generated_text.strip():
            return generated_text
        else:
            print("❌ Hugging Face LLM returned empty response.")
            return None

    except Exception as e:
        print(f"❌ Error generating MCQs: {e}")
        return None

def parse_mcqs(response_text):
    """Parse the LLM response to extract individual questions."""
    if not response_text:
        return []

    questions = []
    # Split by "Question " but be more flexible with parsing
    raw_questions = response_text.strip().split('Question ')[1:]

    for q_text in raw_questions:
        lines = [line.strip() for line in q_text.strip().split('\n') if line.strip()]
        if len(lines) < 6:
            continue
            
        # Extract question (first line after "Question ")
        question_text = lines[0].rstrip(':')  # Remove trailing colon if present
        question_line = f"Question {question_text}"
        
        # Extract options and correct answer
        options = []
        answer_line = ""
        
        for line in lines[1:]:
            if line.startswith(('A)', 'B)', 'C)', 'D)')):
                options.append(line)
            elif "Correct Answer:" in line or "Answer:" in line:
                answer_line = line
                break
        
        # Only add if we have all required components
        if len(options) == 4 and answer_line:
            # Extract just the letter from the answer
            answer_letter = answer_line.split(':')[-1].strip().upper()
            if answer_letter in ['A', 'B', 'C', 'D']:
                questions.append({
                    'question': question_line,
                    'options': options,
                    'correct_answer': answer_letter,
                    'raw_answer_line': answer_line
                })

    return questions

def display_mcqs(questions):
    """Display the generated MCQs in a formatted way."""
    if not questions:
        print("No questions were parsed to display.")
        return
    print("\n" + "="*60)
    print("📝 GENERATED MULTIPLE CHOICE QUESTIONS")
    print("="*60)
    for i, q in enumerate(questions, 1):
        print(f"\n{q.get('question', f'Question {i}')}")
        for option in q.get('options', []):
            print(f"   {option}")
        print(f"   ✅ Correct Answer: {q.get('correct_answer', 'Unknown')}")
        print("-" * 40)

def run_mcq_quiz(questions):
    """Run an interactive CLI-based MCQ quiz."""
    if not questions:
        print("❌ No questions available for the quiz.")
        return
    
    print("\n" + "🎯" * 30)
    print("🎯 STARTING MCQ QUIZ 🎯")
    print("🎯" * 30)
    print(f"📊 Total Questions: {len(questions)}")
    print("📝 Instructions:")
    print("   - Type A, B, C, or D for your answer")
    print("   - Type 'skip' to skip a question")
    print("   - Type 'quit' to exit the quiz")
    print("-" * 60)
    
    score = 0
    answered = 0
    skipped = 0
    
    for i, question in enumerate(questions, 1):
        print(f"\n📋 Question {i}/{len(questions)}:")
        print(f"   {question['question'].replace(f'Question {i}:', '').strip()}")
        print()
        
        # Display options
        for option in question['options']:
            print(f"   {option}")
        
        # Get user answer
        while True:
            user_answer = input(f"\n👉 Your answer (A/B/C/D): ").strip().upper()
            
            if user_answer in ['A', 'B', 'C', 'D']:
                answered += 1
                correct_answer = question['correct_answer']
                
                if user_answer == correct_answer:
                    print("✅ Correct! Well done!")
                    score += 1
                else:
                    print(f"❌ Incorrect. The correct answer is {correct_answer}")
                break
                
            elif user_answer.lower() == 'skip':
                print("⏭️  Question skipped.")
                skipped += 1
                break
                
            elif user_answer.lower() == 'quit':
                print("🚪 Exiting quiz...")
                return display_quiz_results(score, answered, skipped, len(questions))
                
            else:
                print("❌ Invalid input. Please enter A, B, C, D, 'skip', or 'quit'.")
        
        # Show progress
        if i < len(questions):
            print(f"\n📊 Progress: {i}/{len(questions)} | Score: {score}/{answered}")
            input("Press Enter to continue to next question...")
    
    # Display final results
    display_quiz_results(score, answered, skipped, len(questions))

def display_quiz_results(score, answered, skipped, total):
    """Display the final quiz results."""
    print("\n" + "🏆" * 30)
    print("🏆 QUIZ COMPLETED! 🏆")
    print("🏆" * 30)
    
    percentage = (score / answered * 100) if answered > 0 else 0
    
    print(f"📊 RESULTS:")
    print(f"   ✅ Correct Answers: {score}")
    print(f"   ❌ Wrong Answers: {answered - score}")
    print(f"   ⏭️  Skipped: {skipped}")
    print(f"   📝 Total Questions: {total}")
    print(f"   📈 Score: {score}/{answered} ({percentage:.1f}%)")
    
    # Performance feedback
    if percentage >= 90:
        print("🌟 Excellent! Outstanding performance!")
    elif percentage >= 80:
        print("👍 Great job! Well done!")
    elif percentage >= 70:
        print("👌 Good work! Keep it up!")
    elif percentage >= 60:
        print("📈 Not bad! Room for improvement.")
    else:
        print("📚 Consider reviewing the material more.")
    
    print("-" * 60)

def generate_mcqs_from_query(query, num_questions=5, num_chunks=8):
    """Generate MCQs by first querying the ChromaDB, then using HF Inference API."""
    print(f"\nGenerating {num_questions} MCQs for query: '{query}'")
    content, docs = search_database(query, num_chunks)
    if not content:
        print("❌ No relevant content found in ChromaDB.")
        return None
    print(f"✅ Retrieved content from {len(docs)} document chunks")
    print(f"   Content length: {len(content)} characters")
    return generate_mcqs_from_content(content, num_questions)

def main():
    print("=== Quiz Smith MCQ Generator & Quiz Taker ===\n")
    if not test_hf_llm_api():
        print("\n❌ Cannot proceed without a working Hugging Face Inference API connection.")
        return

    while True:
        print("\n" + "="*50)
        print("MAIN MENU")
        print("="*50)
        print("1. Generate MCQs from database query")
        print("2. Take a quiz (requires generated questions)")
        print("3. Exit")
        
        choice = input("\nSelect an option (1-3): ").strip()
        
        if choice == '1':
            # Generate MCQs
            user_query = input("\nEnter your query or topic: ").strip()
            if not user_query:
                print("❌ Please enter a valid query.")
                continue
                
            try:
                num_q_input = input("Number of questions to generate (default: 5): ").strip()
                num_questions = int(num_q_input) if num_q_input else 5
                if num_questions > 10:
                    print("⚠️  Limiting to 10 questions maximum")
                    num_questions = 10
            except ValueError:
                print("Invalid number. Defaulting to 5 questions.")
                num_questions = 5

            mcq_response = generate_mcqs_from_query(user_query, num_questions)
            if mcq_response:
                global generated_questions  # Store for quiz mode
                generated_questions = parse_mcqs(mcq_response)
                if generated_questions:
                    display_mcqs(generated_questions)
                    print(f"\n📊 Successfully generated {len(generated_questions)} questions")
                    print("💡 You can now take a quiz using option 2!")
                else:
                    print("❌ Could not parse questions. Raw output:")
                    print(mcq_response)
            else:
                print("❌ Failed to generate MCQs.")
        
        elif choice == '2':
            # Take quiz
            try:
                if 'generated_questions' in globals() and generated_questions:
                    run_mcq_quiz(generated_questions)
                else:
                    print("❌ No questions available. Please generate MCQs first (option 1).")
            except NameError:
                print("❌ No questions available. Please generate MCQs first (option 1).")
        
        elif choice == '3':
            print("👋 Thank you for using Quiz Smith! Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
