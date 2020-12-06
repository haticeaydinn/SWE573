from django.contrib import admin
from .models import User


# Register your models here.
class LoginAdmin(admin.ModelAdmin):
    list_display = ('name', 'surname', 'email', 'image_url', 'password')


admin.site.register(User, LoginAdmin)
