# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-14 06:32
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('securedpi_facerec', '0001_phot-model'),
    ]

    operations = [
        migrations.AlterField(
            model_name='photo',
            name='image',
            field=models.ImageField(upload_to='training'),
        ),
    ]