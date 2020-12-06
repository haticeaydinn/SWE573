from django.shortcuts import render
from django.http import HttpResponse
from login.models import User


# Create your views here.
def index(request):
    users = User.objects.all()
    # return HttpResponse("Hello SWE world !")
    return render(request, 'index.html', {'users': users})
