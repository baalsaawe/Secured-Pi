# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-12 22:29
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('securedpi_locks', '0007_auto_20161012_2222'),
    ]

    operations = [
        migrations.AlterField(
            model_name='lock',
            name='serial',
            field=models.CharField(max_length=50, unique=True),
        ),
    ]
