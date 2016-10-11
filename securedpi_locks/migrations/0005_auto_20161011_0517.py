# -*- coding: utf-8 -*-
# Generated by Django 1.10.2 on 2016-10-11 05:17
from __future__ import unicode_literals

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('securedpi_locks', '0004_auto_20161010_2142'),
    ]

    operations = [
        migrations.RenameField(
            model_name='lock',
            old_name='is_locked',
            new_name='facial_recognition',
        ),
        migrations.AddField(
            model_name='lock',
            name='is_active',
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name='lock',
            name='status',
            field=models.CharField(choices=[('locked', 'locked'), ('unlocked', 'unlocked'), ('pending', 'pending')], default='unlocked', max_length=8),
        ),
        migrations.AlterField(
            model_name='lock',
            name='description',
            field=models.CharField(blank=True, max_length=25),
        ),
        migrations.AlterField(
            model_name='lock',
            name='location',
            field=models.CharField(max_length=25),
        ),
        migrations.AlterField(
            model_name='lock',
            name='raspberry_pi_id',
            field=models.CharField(max_length=20),
        ),
        migrations.AlterField(
            model_name='lock',
            name='title',
            field=models.CharField(max_length=15),
        ),
        migrations.AlterField(
            model_name='lock',
            name='web_cam_id',
            field=models.CharField(blank=True, max_length=20),
        ),
    ]