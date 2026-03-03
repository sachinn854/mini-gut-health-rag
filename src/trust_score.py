def calculate_trust_score(similarity_scores):
    """
    Calculate trust score from similarity scores.
    
    FAISS returns distance scores where lower = more similar.
    We convert to trust score where higher = more trustworthy.
    
    Methodology:
    1. Normalize each distance score: trust = 1 / (1 + distance)
    2. Average the normalized scores
    3. Result: 0-1 scale where higher = more confident
    
    Args:
        similarity_scores: List of FAISS distance scores (lower = better)
        
    Returns:
        float: Trust score between 0 and 1
    """
    if not similarity_scores:
        return 0.0
    
    # Convert distance to similarity (inverse normalization)
    # Lower distance = higher similarity = higher trust
    normalized_scores = [1 / (1 + score) for score in similarity_scores]
    
    # Average the normalized scores
    trust_score = sum(normalized_scores) / len(normalized_scores)
    
    return trust_score

def get_trust_level(trust_score):
    """
    Convert numerical trust score to human-readable level.
    
    Args:
        trust_score: Float between 0 and 1
        
    Returns:
        str: Trust level description
    """
    if trust_score >= 0.8:
        return "High"
    elif trust_score >= 0.6:
        return "Medium"
    else:
        return "Low"

def explain_trust_score():
    """
    Return explanation of trust score methodology.
    
    Returns:
        str: Explanation text
    """
    explanation = """
Trust Score Methodology:
------------------------
The trust score is calculated as the inverse-normalized average similarity 
distance of the top-3 retrieved chunks from the vector database.

Formula: trust = average(1 / (1 + distance)) for each retrieved chunk

Interpretation:
- Higher semantic alignment between query and retrieved context = higher trust
- Score range: 0.0 to 1.0
- High (≥0.8): Strong semantic match, high confidence
- Medium (0.6-0.8): Moderate match, reasonable confidence  
- Low (<0.6): Weak match, lower confidence

This provides a quantitative measure of how well the retrieved context
matches the user's query, indicating answer reliability.
"""
    return explanation.strip()
