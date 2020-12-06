from django.db import models


# Create your models here.
class User(models.Model):
    name = models.CharField(max_length=255)
    surname = models.CharField(max_length=255)
    email = models.CharField(max_length=255)
    password = models.CharField(max_length=255, default='123456')
    # price = models.FloatField()
    # stock = models.IntegerField()
    image_url = models.CharField(max_length=2083)
