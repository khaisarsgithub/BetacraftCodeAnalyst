from django.db import models

# Project Model
class Project(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name

# Reports Model
class Report(models.Model):
    FREQUENCY_CHOICES = [
        ('Weekly', 'Weekly'),
        ('Bi-Weekly', 'Bi-Weekly')
    ]

    name = models.CharField(max_length=255)
    emails = models.TextField(help_text="Comma-separated list of emails")
    repository_url = models.URLField(max_length=255)
    repository_token = models.CharField(max_length=255)
    prompt = models.TextField()
    active = models.BooleanField(default=True)
    frequency = models.CharField(max_length=10, choices=FREQUENCY_CHOICES)
    project = models.ForeignKey(Project, related_name='reports', on_delete=models.CASCADE)

    def __str__(self):
        return self.name
