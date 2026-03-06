from django.db import models
from django.contrib.auth.models import User

class PredictionLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    session_key = models.CharField(max_length=40, blank=True, null=True)
    user_input = models.TextField()
    predicted_result = models.CharField(max_length=50)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user or self.session_key} - {self.predicted_result}"
