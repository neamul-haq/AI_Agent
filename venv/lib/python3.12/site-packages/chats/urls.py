from django.urls import path
from .views import home, chat_view, new_chat, send_message

app_name = "chats"

urlpatterns = [
    path('', home, name='home'),
    path('new/', new_chat, name='new_chat'),
    path('<str:id>/', chat_view, name='chat'),
    path('<str:id>/send/', send_message, name='send_message'),
]