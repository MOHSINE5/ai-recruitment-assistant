import cv_parser

def test_parsing():
    sample_cv = """
    Jean Dupont
    Email: jean.dupont@gmail.com
    Age: 28 ans
    
    Expérience professionnelle:
    Développeur Python avec 4 ans d'expérience. 
    J'ai travaillé sur des projets en machine learning et data science. 
    Compétences techniques : Python, SQL, Docker. Git et Cloud (AWS).
    
    Formation:
    Master en Informatique (2020)
    """
    
    results = cv_parser.parse_cv_data(sample_cv)
    print("Parsing Results:", results)
    
if __name__ == "__main__":
    test_parsing()
