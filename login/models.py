from django.db import models
from django.contrib.auth.models import User as auth_user_model


# Create your models here.
class CustomUserModel(models.Model):
    user = models.OneToOneField(auth_user_model, on_delete=models.CASCADE)
    image_url = models.CharField(max_length=2083)
