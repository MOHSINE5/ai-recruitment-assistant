import re
import PyPDF2
import docx

def extract_text_from_pdf(file):
    """Extraire le texte d'un fichier PDF."""
    try:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Erreur PDF : {e}")
        return ""

def extract_text_from_docx(file):
    """Extraire le texte d'un fichier DOCX."""
    try:
        doc = docx.Document(file)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        print(f"Erreur DOCX : {e}")
        return ""

def parse_cv_data(text):
    """
    Analyse le texte du CV pour en extraire l'âge, l'expérience, le niveau d'éducation
    et estimer un score de compétences.
    """
    text_lower = text.lower()
    
    # 1. Extraction du nom
    nom = ""
    lignes = [l.strip() for l in text.split('\n') if l.strip()]
    
    # Stratégie 1 : Chercher près de "LinkedIn"
    linkedin_match = re.search(r'LinkedIn:\s*([A-Za-z\s]+)', text)
    if linkedin_match:
        nom = linkedin_match.group(1).split('|')[0].strip()
    
    # Stratégie 2 : Si non trouvé, chercher une ligne avec Nom en MAJUSCULES (nom de famille courant au Maroc/France)
    if not nom:
        for ligne in lignes:
            # Cherche une ligne avec au moins un mot en MAJUSCULES de 3+ lettres (Surnom)
            if re.search(r'\b[A-Z]{3,}\b', ligne) and len(ligne.split()) <= 4:
                nom = ligne
                break
    
    # Stratégie 3 : Hypothèse par défaut (première ligne non vide si courte)
    if not nom and lignes and len(lignes[0].split()) <= 4:
        nom = lignes[0]

    # 2. Extraction de l'email
    email = ""
    email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    if email_match:
        email = email_match.group(0).strip()

    # 3. Extraction de l'âge
    age = 22 # Par défaut pour un étudiant
    age_match = re.search(r'(\d{2})\s*(?:ans|years)', text_lower)
    if age_match:
        age_candidate = int(age_match.group(1))
        if 18 <= age_candidate <= 65:
            age = age_candidate
    
    # 4. Extraction de l'expérience
    experience = 0
    # Chercher les durées de stage/travail (ex: 2 mois, 1 an)
    exp_matches = re.findall(r'(\d+)\s*(?:an|ans|year|years|mois|month|months)', text_lower)
    for val in exp_matches:
        if 'an' in text_lower[text_lower.find(val):text_lower.find(val)+10]:
            experience += int(val)
        else:
            experience += int(val) / 12.0
    
    # Stratégie de secours : si on voit des dates type "2023 - 2024" ou "juillet - août"
    # dans une section expérience, on peut estimer une durée minimale.
    if experience == 0 and re.search(r'exp[eé]rience', text_lower):
        # Si on voit des mois comme Juillet/Août/Septembre etc.
        if re.search(r'juillet|ao[ûu]t|septembre|octobre|novembre|d[eé]cembre|janvier|f[eé]vrier|mars|avril|mai|juin', text_lower):
            experience = 1 # On suppose au moins 1 an cumulé si des mois sont cités
    
    experience = int(round(experience))
    
    # 5. Extraction du niveau d'éducation (Gestion des accents et Bac+X)
    education = 1 # Bac par défaut
    if re.search(r'doctorat|phd|bac\s*\+\s*8', text_lower):
        education = 4
    elif re.search(r'master|ing[eé]nieur|bac\s*[\+\s]*5', text_lower):
        education = 3
    elif re.search(r'licence|bachelor|bac\s*[\+\s]*3|d[eé]veloppement informati(?:que|on)', text_lower):
        education = 2
    elif re.search(r'étudiant en deuxième année|2ème année|bac\s*[\+\s]*2', text_lower):
        education = 2 # Bac+2 souvent assimilé à Licence en cours ou niveau 2
        
    # 6. Score de compétences étendu (Regex Insensible aux accents)
    def clean_accents(s):
        import unicodedata
        return ''.join(c for c in unicodedata.normalize('NFD', s)
                      if unicodedata.category(c) != 'Mn').lower()
    
    text_clean = clean_accents(text)
    
    competences_cles = [
        'python', 'java', 'c++', 'javascript', 'php', 'sql', 'flutter', 'dart', 'html', 'css',
        'laravel', 'react', 'angular', 'node', 'bootstrap', 'django',
        'machine learning', 'data science', 'ai', 'intelligence artificielle', 'ml', 'pandas',
        'git', 'github', 'docker', 'kubernetes', 'aws', 'azure', 'devops', 'cisco',
        'figma', 'canva', 'ui/ux', 'merise', 'uml'
    ]
    
    score_competences = 30
    mots_trouves = 0
    for comp in competences_cles:
        comp_clean = clean_accents(comp)
        if re.search(rf'\b{re.escape(comp_clean)}\b', text_clean):
            mots_trouves += 1
            
    score_competences += mots_trouves * 5
    score_competences = min(100, score_competences)
            
    return {
        'nom': nom,
        'email': email,
        'age': age,
        'experience': experience,
        'education': education,
        'skill_score': score_competences
    }
