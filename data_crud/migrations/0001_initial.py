# Generated by Django 5.1.3 on 2024-12-02 23:23

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Intent',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('tag', models.CharField(max_length=100, unique=True)),
                ('context', models.TextField(blank=True, help_text='Contexto adicional para el intent', null=True)),
            ],
        ),
        migrations.CreateModel(
            name='Pattern',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.CharField(max_length=255)),
                ('intent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='patterns', to='data_crud.intent')),
            ],
        ),
        migrations.CreateModel(
            name='Response',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('text', models.TextField()),
                ('intent', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='responses', to='data_crud.intent')),
            ],
        ),
    ]
