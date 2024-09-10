"""
WSGI config for BetacraftCodeAnalyst project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/5.0/howto/deployment/wsgi/
"""

import os
import sys
from django.core.wsgi import get_wsgi_application
from whitenoise import WhiteNoise
import django


# Add your project directory to the Python path
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BetacraftCodeAnalyst.settings')

print(f"DJANGO_SETTINGS_MODULE: {os.environ.get('DJANGO_SETTINGS_MODULE')}")

django.setup()

application = get_wsgi_application()
application = WhiteNoise(application)
