from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('github/login/', views.github_login, name='github_login'),
    path('github/callback/', views.github_callback, name='github_callback'),
    path('get_repos', views.get_github_repos, name='get_repos'),
    path('analyze_complete_repo', views.analyze_complete_repo, name='analyze_complete_repo'),
    path('get_weekly_report', views.get_weekly_report, name='get_weekly_report'),
    path('analyze_repo', views.analyze_repo_page, name='analyze_repo_page'),
]