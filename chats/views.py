from django.shortcuts import render, redirect, get_object_or_404
from .forms import ChatForm
from .models import ChatSession, Message, IssueReport
from .utils import generate_response
# Create your views here.


def home(request):
    return render(request, 'home.html')


def new_chat(request):
    session = ChatSession.objects.create()
    return redirect('chats:chat', id=str(session.id))


def chat_view(request, id):
    session = get_object_or_404(ChatSession, id=id)

    if request.method == 'POST':
        form = ChatForm(request.POST)
        if form.is_valid():
            user_input = form.cleaned_data['user_input']
            recent_messages = session.messages.order_by('created')
            ai_response, detected_issue = generate_response(user_input, recent_messages, session.id)

            Message.objects.create(session=session, sender='human', text=user_input)
            Message.objects.create(session=session, sender='ai', text=ai_response)

            if detected_issue and detected_issue != "NO_ISSUE":
                # print(detected_issue)
                IssueReport.objects.create(
                    session=session,
                    user_message=user_input,
                    detected_issue=detected_issue.content
                )
            return redirect('chats:chat', id=session.id)
    else:
        form = ChatForm()

    chat_history = session.messages.order_by('created')

    return render(request, 'chat.html', {
        'form': form,
        'chat_history': chat_history,
        'session': session,
    })
