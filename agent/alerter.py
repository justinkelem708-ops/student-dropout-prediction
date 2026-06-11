import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

load_dotenv()

def send_alert(student_id, probability, risk_factors):

    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = os.getenv("EMAIL_RECEIVER")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"ALERTE DECROCHAGE - Eleve {student_id}"
    msg["From"] = sender
    msg["To"] = receiver

    html = f"""
    <html>
    <body style="font-family: Arial; padding: 20px;">
        <h2 style="color: #e74c3c;">ALERTE Decrochage Scolaire</h2>

        <div style="background: #ffeaa7; padding: 15px; border-radius: 8px;">
            <h3>Eleve ID : {student_id}</h3>
            <h3>Probabilite de decrochage : {probability:.1%}</h3>
        </div>

        <h3>Facteurs de risque :</h3>
        <ul>
            {"".join(f"<li>{f}</li>" for f in risk_factors)}
        </ul>

        <div style="background: #dfe6e9; padding: 15px; border-radius: 8px;">
            <h3>Recommandations :</h3>
            <ol>
                <li>Entretien individuel immediat</li>
                <li>Tutorat academique cible</li>
                <li>Suivi assiduite renforce</li>
            </ol>
        </div>

        <p style="color: gray; font-size: 12px;">
            Systeme ML Anti-Decrochage | Developpe par Justin | 2026
        </p>
    </body>
    </html>
    """

    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(sender, password)
            server.sendmail(sender, receiver, msg.as_string())
        print(f"Alerte envoyee pour eleve {student_id}")
        return True
    except Exception as e:
        print(f"Erreur email : {e}")
        return False