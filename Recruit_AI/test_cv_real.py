import cv_parser

def test_on_real_cv():
    with open('cv-test.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    results = cv_parser.parse_cv_data(content)
    print("--- CV TEST RESULTS ---")
    print(f"Nom: {results['nom']}")
    print(f"Email: {results['email']}")
    print(f"Age: {results['age']}")
    print(f"Expérience: {results['experience']}")
    print(f"Education Level: {results['education']}")
    print(f"Skill Score: {results['skill_score']}")
    print("-----------------------")

if __name__ == "__main__":
    test_on_real_cv()
