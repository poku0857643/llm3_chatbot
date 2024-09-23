from django.urls import path
from . import views

urlpatterns = [
    # path('', views.chat_interface, name='chat_interface'),  # URL for the chatbot interface (root path)
    # path('get-response/', views.chatbot_response, name='chatbot_response'),  # URL for chatbot response
    path('get-response/', views.chatbot_response, name='chatbot_response'),
]
