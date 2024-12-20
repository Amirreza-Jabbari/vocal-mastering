# Generated by Django 5.1.4 on 2024-12-10 09:46

import uuid
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("mastering", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="VocalMastering",
            fields=[
                (
                    "id",
                    models.UUIDField(
                        default=uuid.uuid4,
                        editable=False,
                        primary_key=True,
                        serialize=False,
                    ),
                ),
                ("original_audio", models.FileField(upload_to="vocals/")),
                (
                    "mastered_audio",
                    models.FileField(blank=True, null=True, upload_to="mastered/"),
                ),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.DeleteModel(
            name="VocalProcessingJob",
        ),
    ]
