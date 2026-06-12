import arxiv
import os
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_PATH = os.path.join(BASE_DIR, "agent", "agent_log.txt")
REPORT_PATH = os.path.join(BASE_DIR, "agent", "research_report.md")

client = Anthropic()

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def fetch_papers(query="student dropout prediction machine learning", max_results=10):
    log(f"RECHERCHE PAPERS : '{query}'")
    
    client_arxiv = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers = []
    for result in client_arxiv.results(search):
        papers.append({
            "title": result.title,
            "authors": [a.name for a in result.authors[:3]],
            "abstract": result.summary[:500],
            "url": result.entry_id,
            "published": result.published.strftime("%Y-%m-%d")
        })
        log(f"TROUVE : {result.title[:60]}...")
    
    log(f"TOTAL : {len(papers)} papers trouves")
    return papers

def analyze_papers_with_claude(papers):
    log("ANALYSE IA EN COURS...")
    
    papers_text = ""
    for i, p in enumerate(papers, 1):
        papers_text += f"""
Paper {i}:
Titre: {p['title']}
Auteurs: {', '.join(p['authors'])}
Date: {p['published']}
Resume: {p['abstract']}
URL: {p['url']}
---
"""
    
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Tu es un assistant de recherche specialise en ML educatif.

Voici {len(papers)} papers recents sur la prediction du decrochage scolaire :

{papers_text}

Analyse ces papers et produis un rapport structure avec :

1. TENDANCES ACTUELLES (3-5 points)
   - Quelles approches ML sont les plus utilisees ?
   - Quelles donnees sont privilegiees ?

2. GAPS DE RECHERCHE (3-5 points)
   - Qu'est-ce qui manque dans la litterature ?
   - Quels problemes ne sont pas encore resolus ?

3. OPPORTUNITES POUR UN NOUVEAU PAPER (2-3 idees)
   - Angles originaux non explores
   - Comment mon projet (detection precoce sans notes, donnees socio-comportementales, Recall 80.77%) se positionne ?

4. CHERCHEURS A CONTACTER (3-5 auteurs prolifiques)

Reponds en francais."""
        }]
    )
    
    return response.content[0].text

def generate_report(papers, analysis):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    
    report = f"""# Rapport de Veille Scientifique
**Genere automatiquement le {timestamp}**

---

## Papers Analyses ({len(papers)} au total)

"""
    for i, p in enumerate(papers, 1):
        report += f"""### {i}. {p['title']}
- **Auteurs** : {', '.join(p['authors'])}
- **Date** : {p['published']}
- **Resume** : {p['abstract']}
- **Lien** : {p['url']}

"""
    
    report += f"""---

## Analyse IA

{analysis}

---

*Rapport genere par Agent de Recherche ML | Projet Justin | 2026*
"""
    
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    
    log(f"RAPPORT SAUVEGARDE : {REPORT_PATH}")
    return report

def run_researcher():
    log("="*50)
    log("AGENT RECHERCHE DEMARRE")
    log("="*50)
    
    papers = fetch_papers(
        query="student dropout prediction machine learning early detection",
        max_results=10
    )
    
    if not papers:
        log("ERREUR : Aucun paper trouve")
        return
    
    analysis = analyze_papers_with_claude(papers)
    generate_report(papers, analysis)
    
    log("="*50)
    log("RAPPORT GENERE AVEC SUCCES")
    log(f"Fichier : agent/research_report.md")
    log("="*50)
    
    print("\n" + "="*60)
    print("ANALYSE IA :")
    print("="*60)
    print(analysis)

if __name__ == "__main__":
    run_researcher()