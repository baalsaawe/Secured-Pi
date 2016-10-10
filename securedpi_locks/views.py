from django.shortcuts import render
from django.views.generic import TemplateView
from securedpi_locks.models import Lock
from securedpi_locks.serializers import LockSerializer
from rest_framework import generics
# from django.contrib.auth.models import User
from rest_framework import permissions
from securedpi_locks.permissions import IsOwnerOrReadOnly
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse
from rest_framework import renderers
from rest_framework import viewsets
from rest_framework.decorators import detail_route


class DashboardView(TemplateView):
    """Establish class for Dashboard page view."""
    template_name = 'securedpi_locks/dashboard.html'

    def get_context_data(self, **kwargs):
        context = super(DashboardView, self).get_context_data(**kwargs)
        current_user = self.request.user
        locks_by_created_date = current_user.locks.order_by('-date_created')
        context['locks'] = locks_by_created_date
        return context


class LockViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides 'list', 'create', 'retrieve',
    'update' and 'destroy' actions.
    Additionally we also provide an extra 'highlight' action.
    """
    queryset = Lock.objects.all()
    serializer_class = LockSerializer
    permission_classes = (permissions.IsAuthenticatedOrReadOnly,
                          IsOwnerOrReadOnly,)

    @detail_route(renderer_classes=[renderers.StaticHTMLRenderer])
    def highlight(self, request, *args, **kwargs):
        lock = self.get_object()
        return Response(lock.highlighted)

    def perform_create(self, serializer):
        serializer.save(user=self.request.user)
