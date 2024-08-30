from __future__ import print_function
import time
from django.http import HttpResponse
from django.shortcuts import render
import os
from dotenv import load_dotenv
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import datetime
import sib_api_v3_sdk
from sib_api_v3_sdk.rest import ApiException
from pprint import pprint
import threading
# from git_app.views import analyze_repo

import schedule

# Load the API key from environment variables
load_dotenv()

# Brevo Configurations
configuration = sib_api_v3_sdk.Configuration()
configuration.api_key['api-key'] = os.environ.get('BREVO_API_KEY')

api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
            

# Send Email using Brevo
def send_brevo_mail(subject, html_content, emails):
    api_instance = sib_api_v3_sdk.TransactionalEmailsApi(sib_api_v3_sdk.ApiClient(configuration))
    # subject = "My Subject"
    # html_content = "<html><body><h1>This is my first transactional email </h1></body></html>"
    if isinstance(emails, str):
        emails = [{"email":email.strip(), "name":email.split("@")[0]} for email in emails.split(',')]
    print(f"Number of emails: {len(emails)}")
    
    # # Create a list of dictionaries for the 'to' parameter
    # to = [{"email": email, "name": email.split("@")[0]} for email in emails]
    to = emails
    cc = [{"email":"mdkhaisars118@gmail.com", "name":"Mohammed Khaisar"}]
    # bcc = [{}]
    # reply_to = {}
    sender = {"name":"Mohammed Khaisar", "email":"khaisar@betacraft.io"}
    headers = {"Some-Custom-Name":"unique-id-1"}
    params = {"parameter":"My param value","subject":"New Subject"}
    # for email in emails:
    # to = [{"email":email, "name":email.split("@")[0]}]
    print(f"To: {to}")
    try:
        send_smtp_email = sib_api_v3_sdk.SendSmtpEmail(to=to, cc=cc, headers=headers, html_content=html_content, sender=sender, subject=subject)
        api_response = api_instance.send_transac_email(send_smtp_email)
        print(f"Email sent successfully: {api_response}")
        return True, "Email sent successfully"
    except Exception as e:
        print(f"Unexpected error when sending email: {e}")
        return False, f"An unexpected error occurred while sending the email {e}"

# Create your views here.
def send_email(subject, body, to_email):
    # Your email credentials
    from_email = os.environ.get('EMAIL_ADDRESS')
    password = os.environ.get('EMAIL_PASSWORD')

    # Create the email content
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'html'))

    # Connect to the Gmail SMTP server
    try:
        server = smtplib.SMTP(os.environ.get("EMAIL_SERVER"), 587)
        server.starttls()  # Upgrade the connection to a secure encrypted SSL/TLS connection
        server.login(from_email, password)
        text = msg.as_string()
        server.sendmail(from_email, to_email, text)
        server.quit()
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email. Error: {str(e)}")


def weekly_job(repo_name, report, email):    
    last_week = datetime.datetime.now() - datetime.timedelta(weeks=1)
    today = datetime.datetime.now()
    emails = email.split(',')
    
    # Analyze the repository
    

    # Schedule the job to run every week
    schedule.every().friday.at("18:00").do(send_brevo_mail, 
                        subject=f"{repo_name} : {str(last_week)[:10]} - {str(today)[:10]}", 
                        html_content=report, 
                        to=email)
    # Create a separate thread for the scheduler
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # This makes the thread exit when the main program exits
    scheduler_thread.start()
    print("Job Scheduled Successfully")
    
    
def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(60*60*24*7)
        print(f"Email sent Successfully")