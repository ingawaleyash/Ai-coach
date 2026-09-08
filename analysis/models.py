from django.db import models
from django.contrib.auth.models import User


class Player(models.Model):
    name = models.CharField(max_length=100)
    age = models.IntegerField()
    batting_style = models.CharField(max_length=50)

    def __str__(self):
        return self.name


class UploadedVideo(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=200)
    video = models.FileField(upload_to="videos/")
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title