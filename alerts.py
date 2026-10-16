# alerts.py
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from twilio.rest import Client


# -------------------------------------
# CONFIGURATION
# -------------------------------------

# Email Configuration
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"    # App Password (not login password)
EMAIL_RECEIVER = "receiver_email@gmail.com"

# WhatsApp Configuration (Twilio)
TWILIO_SID = "your_twilio_sid"
TWILIO_AUTH = "your_twilio_auth"
TWILIO_WHATSAPP = "whatsapp:+14155238886"   # Twilio Sandbox Number
ADMIN_WHATSAPP = "whatsapp:+91XXXXXXXXXX"  # Your WhatsApp number


# -------------------------------------
# EMAIL ALERT
# -------------------------------------

def send_email_alert(subject, message):

    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        msg["Subject"] = subject

        msg.attach(MIMEText(message, "plain"))

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()

        print("Email Alert Sent ✔")

    except Exception as e:
        print("Email Send Error:", e)


# -------------------------------------
# WHATSAPP ALERT
# -------------------------------------

def send_whatsapp_alert(message):

    try:
        client = Client(TWILIO_SID, TWILIO_AUTH)

        client.messages.create(
            body=message,
            from_=TWILIO_WHATSAPP,
            to=ADMIN_WHATSAPP
        )

        print("WhatsApp Alert Sent ✔")

    except Exception as e:
        print("WhatsApp Send Error:", e)


# -------------------------------------
# MAIN ALERT WRAPPER
# -------------------------------------

def send_alert(plate, violation_type, image_path, extra_info=""):

    message = (
        f"🚨 TRAFFIC VIOLATION DETECTED 🚨\n\n"
        f"Type: {violation_type}\n"
        f"Plate: {plate}\n"
        f"Details: {extra_info}\n"
        f"Image: {image_path}\n"
    )

    # send both email & WhatsApp
    send_email_alert(f"Traffic Violation - {violation_type}", message)
    send_whatsapp_alert(message)
