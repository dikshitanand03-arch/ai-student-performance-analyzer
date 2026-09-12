"""
src/ai_explainer.py
-------------------
AI explanation and recommendation engine module using OpenAI API,
with automatic fallback to a structured rule-based heuristic engine.
"""

import os
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

def generate_heuristic_explanation(
    student_dict: Dict[str, Any],
    predicted_mark: float,
    risk_level: str,
    contributing_factors: List[str]
) -> str:
    """Fallback rule-based natural language generator when no OpenAI API key is configured."""
    student_id = student_dict.get("student_id", "Student")
    attendance = student_dict.get("attendance", 0)
    study_hours = student_dict.get("study_hours", 0)
    prev_marks = student_dict.get("previous_marks", 0)

    if risk_level == "HIGH RISK":
        primary_reasons = []
        if attendance < 70:
            primary_reasons.append(f"low attendance ({attendance:.1f}%)")
        if study_hours < 8:
            primary_reasons.append(f"insufficient weekly study hours ({study_hours:.1f} hrs/week)")
        if prev_marks < 50:
            primary_reasons.append(f"weak baseline academic marks ({prev_marks:.1f}/100)")
            
        reason_str = ", ".join(primary_reasons) if primary_reasons else "multiple low engagement metrics"
        
        explanation = (
            f"Student #{student_id} is flagged as **HIGH RISK** with a predicted final mark of **{predicted_mark:.1f}/100**. "
            f"The primary contributing factors are {reason_str}. "
            f"**Recommended Action**: Urgent intervention required. Prioritize increasing lecture attendance above 80% "
            f"and establishing a structured study routine of at least 10–12 hours per week."
        )
    elif risk_level == "MODERATE RISK":
        explanation = (
            f"Student #{student_id} is flagged as **MODERATE RISK** (Predicted Final Mark: **{predicted_mark:.1f}/100**). "
            f"While current performance is acceptable, key areas like attendance ({attendance:.1f}%) and assignment consistency "
            f"warrant monitoring. **Recommended Action**: Encourage weekly study groups and target a 5% improvement in assignment scores."
        )
    else:
        explanation = (
            f"Student #{student_id} is on track (**SAFE / LOW RISK**) with a strong predicted mark of **{predicted_mark:.1f}/100**. "
            f"They exhibit healthy study habits ({study_hours:.1f} study hrs/week) and attendance ({attendance:.1f}%). "
            f"**Recommended Action**: Maintain current study schedule and consider advanced/enrichment materials."
        )
        
    return explanation

def generate_ai_explanation(
    student_dict: Dict[str, Any],
    prediction_result: Dict[str, Any],
    api_key: str = None
) -> str:
    """
    Generates personalized feedback using OpenAI GPT API.
    Gracefully falls back to heuristic generation if API key is missing or call fails.
    """
    key = api_key or os.getenv("OPENAI_API_KEY")
    
    predicted_mark = prediction_result.get("predicted_final_marks", 0.0)
    risk_level = prediction_result.get("risk_level", "UNKNOWN")
    risk_prob = prediction_result.get("risk_probability", 0.0)
    factors = prediction_result.get("contributing_factors", [])
    
    # Check if key is absent or placeholder template string
    if not key or key == "your_openai_api_key_here":
        return generate_heuristic_explanation(student_dict, predicted_mark, risk_level, factors)
        
    try:
        import openai
        client = openai.OpenAI(api_key=key)
        
        prompt = f"""
You are an expert AI Academic Mentor. Analyze the following student metrics and risk model prediction:

- Student ID: {student_dict.get('student_id', 'N/A')}
- Attendance: {student_dict.get('attendance')}%
- Weekly Study Hours: {student_dict.get('study_hours')} hrs
- Previous Marks: {student_dict.get('previous_marks')}/100
- Assignment Score: {student_dict.get('assignment_score')}/100
- Sleep Hours: {student_dict.get('sleep_hours')} hrs/day
- Participation: {student_dict.get('participation')}/5
- ML Predicted Mark: {predicted_mark}/100
- Predicted Risk Level: {risk_level} (Probability: {risk_prob}%)
- Identified Risk Factors: {', '.join(factors)}

Provide a concise (3-4 sentences max), professional, empathetic, and actionable advisory note for the student and their tutor.
1. State their risk status and main driver.
2. Provide 2 specific, realistic actionable steps for improvement.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and concise academic advisor AI."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=180,
            temperature=0.7
        )
        
        ai_text = response.choices[0].message.content.strip()
        return f"**[AI Explanation (OpenAI)]**\n\n{ai_text}"
        
    except Exception as e:
        # Fall back gracefully on network or API key error
        fallback_text = generate_heuristic_explanation(student_dict, predicted_mark, risk_level, factors)
        return f"{fallback_text}\n\n*(Note: OpenAI API call skipped/failed: {str(e)})*"
