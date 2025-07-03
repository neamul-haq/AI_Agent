from django.contrib import admin
from chats.models import ChatSession, Message, IssueReport
# Register your models here.
admin.site.register(ChatSession)
admin.site.register(Message)
admin.site.register(IssueReport)