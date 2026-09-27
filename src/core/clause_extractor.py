import re

# Define important legal clause keywords and regex patterns
CLAUSE_PATTERNS = {
    "Termination": r"(?i)(termination|end of agreement|cancel|expiry)",
    "Confidentiality": r"(?i)(confidentiality|non-disclosure|NDA|secret)",
    "Governing Law / Jurisdiction": r"(?i)(governing law|jurisdiction|court|venue)",
    "Payment Terms": r"(?i)(payment|fees|consideration|invoice|charges)",
    "Liability": r"(?i)(liability|damages|responsibility|loss)",
    "Indemnity": r"(?i)(indemnity|indemnification|hold harmless)",
}

def extract_clauses(text):
    """Extract key clauses from document text."""
    clauses_found = {}

    # Split text into paragraphs
    paragraphs = re.split(r'\n{2,}|\r\n{2,}', text)

    for name, pattern in CLAUSE_PATTERNS.items():
        matched = []
        for para in paragraphs:
            if re.search(pattern, para):
                matched.append(para.strip())
        if matched:
            clauses_found[name] = matched

    return clauses_found
