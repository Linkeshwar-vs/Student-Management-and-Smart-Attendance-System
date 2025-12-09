# services/quiz_generation.py
import os
import json
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.chat_models import ChatOpenAI
from .utils.quiz_utils import extract_text_from_file

# Set env to point to OpenRouter if needed:
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
os.environ["OPENAI_API_KEY"] = "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"

# JSON schema we expect:
schema = {
    "type": "object",
    "properties": {
        "quiz_title": {"type": "string"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stem": {"type": "string"},
                    "options": {"type": "array", "minItems": 4, "maxItems": 4, "items": {"type": "string"}},
                    "answer_index": {"type": "integer", "minimum": 0, "maximum": 3},
                    "difficulty": {"type": "string"},
                    "explanation": {"type": "string"}
                },
                "required": ["stem", "options", "answer_index"]
            }
        }
    },
    "required": ["questions"]
}

# Updated parser for latest LangChain
output_parser = JsonOutputParser(schema=schema, key=None)  # 'key' replaces old JsonOutputKey

prompt_template = """You are a helpful question-authoring assistant for teachers.
Generate a multiple-choice quiz (each question must have exactly 4 options). 
Return output as strict JSON matching this schema: {schema}.

Context from provided teaching material (use to create questions):
{context}

Teacher prompt: {teacher_prompt}
Number of questions: {num_questions}
Difficulty: {difficulty}

Produce clear, unambiguous stems. Put the correct option index (0..3) in 'answer_index'. If you cannot create enough Qs, fill remaining with best-effort general knowledge Qs related to the context.
"""

prompt = PromptTemplate(
    input_variables=["schema", "context", "teacher_prompt", "num_questions", "difficulty"],
    template=prompt_template
)

def generate_quiz_from_file(filepath, teacher_prompt="Generate Quiz", num_questions=10, difficulty="Medium"):
    context = extract_text_from_file(filepath)
    # Optional truncation if context is very large
    if len(context) > 30000:
        context = context[:30000]
    
    model = ChatOpenAI(temperature=0.2, max_tokens=1500, model="nousresearch/deephermes-3-llama-3-8b-preview:free")  # adjust to your model
    chain = LLMChain(llm=model, prompt=prompt)
    
    raw = chain.run(
        schema=json.dumps(schema),
        context=context,
        teacher_prompt=teacher_prompt,
        num_questions=num_questions,
        difficulty=difficulty
    )
    
    parsed = output_parser.parse(raw)
    return parsed  # Returns dict with quiz structure
