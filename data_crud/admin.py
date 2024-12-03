from django.contrib import admin
from .models import Intent, Pattern, Response

class PatternInline(admin.TabularInline):
    model = Pattern
    extra = 1  # Número de filas adicionales para agregar patrones directamente

class ResponseInline(admin.TabularInline):
    model = Response
    extra = 1  # Número de filas adicionales para agregar respuestas directamente

@admin.register(Intent)
class IntentAdmin(admin.ModelAdmin):
    list_display = ("tag",)
    search_fields = ("tag", "context")
    inlines = [PatternInline, ResponseInline]  # Permite gestionar patrones y respuestas en la misma página

# También puedes registrar Pattern y Response por separado si es necesario
# admin.site.register(Pattern)
# admin.site.register(Response)
