from django.db import models

class Intent(models.Model):
    tag = models.CharField(max_length=100, unique=True)
    context = models.TextField(blank=True, null=True, help_text="Contexto adicional para el intent")

    def __str__(self):
        return self.tag

class Pattern(models.Model):
    intent = models.ForeignKey(Intent, on_delete=models.CASCADE, related_name="patterns")
    text = models.CharField(max_length=255)

    def __str__(self):
        return self.text

class Response(models.Model):
    intent = models.ForeignKey(Intent, on_delete=models.CASCADE, related_name="responses")
    text = models.TextField()

    def __str__(self):
        return self.text
