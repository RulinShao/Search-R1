import re
from collections import Counter
import string


def compute_score(solution_str, ground_truth, format_score=0., score=0.) -> float:
    """
    Compute score for GPQA solutions.
    
    Args:
        solution_str: The model's generated solution string
        ground_truth: The correct answer
        
    Returns:
        float: Score between 0.0 and 1.0
    """
    answer_str = ground_truth['target'][0]
    print(f"🩷💛🩵 Computing gpqa scores:\n")
    print(f"🩷💛🩵 {solution_str}\n{answer_str}")
    retval = 0.0
    try:
        # Extract the answer from the solution string
        pred_answer = extract_answer(solution_str, mode='choose')
        
        # Check if the extracted answer is empty
        if pred_answer == '':
            return retval
        
        # For GPQA multiple choice, check if answer is a single character
        if len(pred_answer) > 1:
            return retval
        
        # Normalize the predicted and ground truth answers
        normalized_pred_answer = normalize_answer(pred_answer)
        normalized_ground_truth = normalize_answer(answer_str)
        
        # Check exact match
        if normalized_pred_answer == normalized_ground_truth:
            retval = 1.0
        # else:
        #     retval = 0.1  # give a bit format score
    except Exception as e:
        print(f"Error in compute_score: {e}")
    
    print(f"🩷💛🩵 Correctness: {retval}")
    return retval


def extract_answer(output, mode='choose'):
    """
    Extract answer from the model output.
    
    Args:
        output: The model's output string
        mode: The evaluation mode ('choose' for GPQA)
        
    Returns:
        str: The extracted answer
    """
    extracted_text = ''
    
    # Extraction logic for 'choose' mode (GPQA)
    pattern = r'\\boxed\{(.*)\}'
    matches = re.findall(pattern, output)
    if matches:
        extracted_text = matches[-1]  # Take the last match
        # Handle 'choose' mode
        inner_pattern = r'\\text\{(.*)\}'
        inner_matches = re.findall(inner_pattern, extracted_text)
        if inner_matches:
            extracted_text = inner_matches[-1]  # Take the last match
        extracted_text = extracted_text.strip("()")
    
    # If no \boxed matches found, look for other patterns common in GPQA answers
    if not extracted_text:
        # Look for a pattern like "The answer is A" or "I choose B"
        choice_patterns = [
            r'(?:the answer is|i choose|answer:|choice:|option:)\s*([A-Ea-e])',
            r'(?:the correct answer is|the answer choice is | <answer>)\s*([A-Ea-e])'
        ]
        
        for pattern in choice_patterns:
            matches = re.findall(pattern, output.lower())
            if matches:
                extracted_text = matches[-1].upper()  # Take the last match and make uppercase
                break
                
    return extracted_text


def normalize_answer(text):
    """
    Normalize the answer string to facilitate comparison.
    
    Args:
        text: The text to normalize
        
    Returns:
        str: The normalized text
    """
    text = text.lower()
    text = " ".join(text.strip().split())
    return text


def evaluate_predictions(output, labeled_answer, mode='choose'):
    """
    Evaluate a single prediction.
    
    Args:
        output: The model's output
        labeled_answer: The ground truth answer
        mode: Evaluation mode ('choose' for GPQA)
        
    Returns:
        dict: Metrics for the prediction
        str: The extracted answer
    """
    final_metric = {"is_valid_answer": False, "acc": 0, "em": 0, "f1": 0}
    pred_answer = extract_answer(output, mode=mode)
    
    if pred_answer != '':
        final_metric["is_valid_answer"] = True
    
    normalized_pred_answer = normalize_answer(pred_answer)
    normalized_ground_truth = normalize_answer(labeled_answer)

    em = int(normalized_pred_answer == normalized_ground_truth)
    acc = int(normalized_ground_truth in normalized_pred_answer)

    prediction_tokens = normalized_pred_answer.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        f1 = 0
    else:
        precision = 1.0 * num_same / len(prediction_tokens) if len(prediction_tokens) > 0 else 0
        recall = 1.0 * num_same / len(ground_truth_tokens) if len(ground_truth_tokens) > 0 else 0
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

    final_metric["em"] = em
    final_metric["acc"] = acc
    final_metric["f1"] = f1
    
    return final_metric, pred_answer


if __name__ == "__main__":
    # Evaluate the predictions
    solution_str = "To solve this problem, we need to analyze the given potential and match it with the correct energy spectrum formula.\n\n1. **Identify the Potential Form:**\n   The given potential is \\( V(r, \\theta) = \\frac{1}{2} kr^2 + \\frac{3}{2} kr^2 \\cos^2(\\theta) \\).\n\n2. **Separation of Variables:**\n   This potential can be separated into radial and angular parts. The angular part involves \\(\\cos^2(\\theta)\\), which suggests a harmonic oscillator potential in polar coordinates.\n\n3. **Radial and Angular Equations:**\n   - The radial part of the potential is \\(\\frac{1}{2} kr^2\\), which is a simple harmonic oscillator.\n   - The angular part is \\(\\frac{3}{2} kr^2 \\cos^2(\\theta)\\), which can be rewritten using the identity \\(\\cos^2(\\theta) = \\frac{1 + \\cos(2\\theta)}{2}\\).\n\n4. **Energy Spectrum Form:**\n   For a 2D harmonic oscillator, the energy levels are given by:\n   \\[\n   E = \\hbar \\omega (n_x + n_y + \\frac{1}{2})\n   \\]\n   where \\(\\omega = \\sqrt{\\frac{k}{m}}\\).\n\n5. **Matching the Given Options:**\n   - Option (A): \\(E = (3n_x + 2n_y + \\frac{1}{2}) \\hbar \\sqrt{\\frac{k}{m}}\\)\n   - Option (B): \\(E = (2n_x + n_y + \\frac{3}{2}) \\hbar \\sqrt{\\frac{k}{m}}\\)\n   - Option (C): \\(E = (2n_x + 3n_y + \\frac{1}{2}) \\hbar \\sqrt{\\frac{k}{m}}\\)\n   - Option (D): \\(E = (n_x + 3n_y + \\frac{3}{2}) \\hbar \\sqrt{\\frac{k}{m}}\\)\n\n   The correct form should match the standard 2D harmonic oscillator energy spectrum, which is \\(E = \\hbar \\omega (n_x + n_y + \\frac{1}{2})\\).\n\n6. **Conclusion:**\n   The correct option is the one that matches the standard form, which is:\n   \\[\n   E = (n_x + n_y + \\frac{1}{2}) \\hbar \\sqrt{\\frac{k}{m}}\n   \\]\n   Comparing this with the given options, we see that the closest match is:\n   \\[\n   E = (n_x + 3n_y + \\frac{3}{2}) \\hbar \\sqrt{\\frac{k}{m}}\n   \\]\n   This is the closest to the standard form, but it has an extra \\(2n_y\\) term and an extra \\(\\frac{1}{2}\\) term, which suggests a modified form.\n\nGiven the options, the closest and most plausible match is:\n\\boxed{D}"
    ground_truth = "D"
    score = compute_score(solution_str, ground_truth)
    print(f"Score: {score}")