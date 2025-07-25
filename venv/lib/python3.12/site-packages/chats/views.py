from django.shortcuts import render, redirect, get_object_or_404
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import json
from .forms import ChatForm
from .models import ChatSession, Message, IssueReport
from .utils import generate_response

def home(request):
    return render(request, 'home.html')

def new_chat(request):
    session = ChatSession.objects.create()
    return redirect('chats:chat', id=str(session.id))

def chat_view(request, id):
    session = get_object_or_404(ChatSession, id=id)
    
    if request.method == 'GET':
        form = ChatForm()
        chat_history = session.messages.order_by('created')
        
        return render(request, 'chat.html', {
            'form': form,
            'chat_history': chat_history,
            'session': session,
        })

@csrf_exempt
@require_http_methods(["POST"])
def send_message(request, id):
    session = get_object_or_404(ChatSession, id=id)
    
    try:
        data = json.loads(request.body)
        user_input = data.get('message', '').strip()
        
        if not user_input:
            return JsonResponse({'error': 'Message cannot be empty'}, status=400)
        
        # Get recent messages for context
        recent_messages = session.messages.order_by('created')
        
        # Generate AI response
        ai_response, detected_issue = generate_response(user_input, recent_messages, session.id)
        
        # Save messages to database
        user_message = Message.objects.create(session=session, sender='human', text=user_input)
        ai_message = Message.objects.create(session=session, sender='ai', text=ai_response)
        
        # Handle issue detection
        if detected_issue and detected_issue != "NO_ISSUE":
            IssueReport.objects.create(
                session=session,
                user_message=user_input,
                detected_issue=detected_issue.content
            )
        
        return JsonResponse({
            'success': True,
            'user_message': {
                'id': user_message.id,
                'text': user_message.text,
                'sender': user_message.sender,
                'created': user_message.created.isoformat()
            },
            'ai_message': {
                'id': ai_message.id,
                'text': ai_message.text,
                'sender': ai_message.sender,
                'created': ai_message.created.isoformat()
            }
        })
        
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)