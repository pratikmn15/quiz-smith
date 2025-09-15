# Generate MCQs using HuggingFace Inference API and save to JSON
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from query_database import search_database

# Load environment variables
load_dotenv()
HF_LLM_API_KEY = os.getenv("HF_LLM_API_KEY")
MODEL_ID = "Qwen/Qwen3-14B"

def setup_llm():
    """Initialize the Hugging Face InferenceClient for generation."""
    print("Setting up Hugging Face InferenceClient for question generation...")
    if not HF_LLM_API_KEY:
        print("❌ HF_LLM_API_KEY not found in environment variables.")
        print("Please add your HuggingFace API key to the .env file: HF_LLM_API_KEY=your_hf_api_key_here")
        return None
    
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
        max_content_length = 3000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "..."
            print(f"⚠️  Content truncated to {max_content_length} characters for model processing")

        prompt = create_mcq_prompt(content, num_questions)
        print("🤖 Generating questions with HuggingFace Inference API...")

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
    raw_questions = response_text.strip().split('Question ')[1:]

    for q_text in raw_questions:
        lines = [line.strip() for line in q_text.strip().split('\n') if line.strip()]
        if len(lines) < 6:
            continue
            
        # Extract question (first line after "Question ")
        question_text = lines[0].rstrip(':')
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

def save_mcqs_to_json(questions, query, filename=None):
    """Save the generated MCQs to a JSON file."""
    if not questions:
        print("❌ No questions to save.")
        return None
    
    # Generate filename if not provided
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = "".join(c for c in query if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_query = safe_query.replace(' ', '_')[:30]  # Limit length
        filename = f"mcqs_{safe_query}_{timestamp}.json"
    
    # Create the data structure
    mcq_data = {
        "metadata": {
            "query": query,
            "generated_at": datetime.now().isoformat(),
            "total_questions": len(questions),
            "model_used": MODEL_ID
        },
        "questions": []
    }
    
    # Process each question
    for i, q in enumerate(questions, 1):
        question_data = {
            "id": i,
            "question": q['question'],
            "options": {
                "A": q['options'][0].replace('A) ', ''),
                "B": q['options'][1].replace('B) ', ''),
                "C": q['options'][2].replace('C) ', ''),
                "D": q['options'][3].replace('D) ', '')
            },
            "correct_answer": q['correct_answer']
        }
        mcq_data["questions"].append(question_data)
    
    # Save to JSON file
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(mcq_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ MCQs saved to: {filename}")
        print(f"📊 Total questions saved: {len(questions)}")
        return filename
        
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")
        return None

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
    """Main function for MCQ generation."""
    print("=== Quiz Smith MCQ Generator ===\n")
    
    if not test_hf_llm_api():
        print("\n❌ Cannot proceed without a working Hugging Face Inference API connection.")
        return

    while True:
        print("\n" + "="*50)
        print("MCQ GENERATOR")
        print("="*50)
        
        # Get user input
        user_query = input("\nEnter your query or topic (or 'quit' to exit): ").strip()
        
        if user_query.lower() in ['quit', 'exit', 'q']:
            print("👋 Thank you for using Quiz Smith! Goodbye!")
            break
            
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

        # Generate MCQs
        mcq_response = generate_mcqs_from_query(user_query, num_questions)
        if mcq_response:
            questions = parse_mcqs(mcq_response)
            if questions:
                # Display questions
                display_mcqs(questions)
                
                # Save to JSON
                saved_file = save_mcqs_to_json(questions, user_query)
                
                if saved_file:
                    print(f"\n💾 Questions saved to: {saved_file}")
                    print("🎯 You can now use this JSON file for web interface or other applications!")
                else:
                    print("❌ Failed to save questions to JSON file.")
                    
            else:
                print("❌ Could not parse questions. Raw output:")
                print(mcq_response)
        else:
            print("❌ Failed to generate MCQs.")

if __name__ == "__main__":
    main()
