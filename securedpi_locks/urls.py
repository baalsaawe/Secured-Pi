from django.conf.urls import url, include
from django.views.generic import TemplateView, DetailView
from django.contrib.auth.decorators import login_required
from securedpi_locks import views
from securedpi_locks.models import Lock


urlpatterns = [
    # url(r'^(?P<pk>\d+)/$',
    #     login_required(DetailView.as_view(
    #         template_name='securedpi_locks/lock_details.html',
    #         model=Lock,
    #         context_object_name='lock'
    #     )),
    #     name='lock_details'),
    # url(r'^manual/unlock/(?P<pk>\d+)/$',
    #     login_required(views.manual_unlock),
    #     name='manual_unlock'),
    # url(r'^manual/lock/(?P<pk>\d+)/$',
    #     login_required(views.manual_lock),
    #     name='manual_lock'),
    url(r'^manual-unlock/(?P<pk>\d+)/$',
        login_required(views.manual_action),
        kwargs={'action': 'unlock'},
        name='manual_unlock'),
    url(r'^manual-lock/(?P<pk>\d+)/$',
        login_required(views.manual_action),
        kwargs={'action': 'lock'},
        name='manual_lock'),
]


urlpatterns += [
    url(r'^api-auth/', include('rest_framework.urls',
                               namespace='rest_framework')),
]
