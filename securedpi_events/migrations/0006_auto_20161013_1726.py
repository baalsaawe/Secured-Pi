# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-13 17:26
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('securedpi_events', '0005_auto_20161013_0559'),
    ]

    operations = [
        migrations.AlterField(
            model_name='event',
            name='status',
            field=models.CharField(default='failed', max_length=20),
        ),
    ]
