from django.conf.urls import url, include
from django.views.generic import TemplateView, DetailView
from django.contrib.auth.decorators import login_required
from securedpi_events import views


urlpatterns = [
    url(r'^(?P<pk>\d+)/$',
        login_required(views.EventView.as_view()),
        name='events'),
    url(r'^(?P<pk>\d+)/delete-old-events/$',
        login_required(views.delete_old_events),
        name='delete_old_events'),
]
