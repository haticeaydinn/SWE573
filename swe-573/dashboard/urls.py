from django.urls import path
from . import views

urlpatterns = [
    path('', views.index),
    path('image/', views.mplimage),
    path('displayposts/', views.display_text),
    path('tagme/', views.tagme_result),
    path('sentiment/', views.sentiment_analysis),
    path('wordcloud/', views.wordcloud_img),
    path('entitylist/', views.entity_list),
    path('networkgraph/', views.word_co_networkg),
    path('showhistory/', views.show_history),
    path('displaynetworkg/', views.network_graph)

]
