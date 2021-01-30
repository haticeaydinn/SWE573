from django.db import models


# Create your models here.
class History(models.Model):
    user_id = models.CharField(max_length=45)
    date = models.CharField(max_length=20)
    search_word = models.CharField(max_length=50)
    search_date_interval = models.CharField(max_length=20)


class PostTitle(models.Model):
    user_id = models.CharField(max_length=45)
    date = models.CharField(max_length=20)
    search_word = models.CharField(max_length=50)
    search_date_interval = models.CharField(max_length=20)
    title = models.CharField(max_length=2083)