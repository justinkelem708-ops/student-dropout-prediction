import schedule
import time
from agent import run_agent, log
from evaluator import retrain_if_needed

def weekly_routine():
    """Routine hebdomadaire : evaluation + alertes."""
    log("="*50)
    log("ROUTINE HEBDOMADAIRE DEMARREE")
    log("="*50)
    
    # 1. Evaluer et reentainer si necessaire
    retrain_if_needed(min_recall=0.75)
    
    # 2. Analyser les nouveaux eleves
    run_agent()
    
    log("ROUTINE HEBDOMADAIRE TERMINEE")

def start_monitor():
    log("SURVEILLANCE AUTOMATIQUE ACTIVEE")
    log("Routine programmee : chaque lundi a 08h00")

    schedule.every().monday.at("08:00").do(weekly_routine)

    # TEST : decommenter pour tester toutes les 2 minutes
    # schedule.every(2).minutes.do(weekly_routine)

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    weekly_routine()
    start_monitor()