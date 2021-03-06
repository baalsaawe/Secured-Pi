from securedpi_events.models import Event
from securedpi_locks.models import Lock
from securedpi_api.serializers import EventSerializer, LockSerializer
from rest_framework.response import Response
from rest_framework import renderers
from rest_framework import viewsets
from rest_framework.decorators import detail_route
from django.shortcuts import get_object_or_404


class LockViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides some actions.

    Provides 'list', 'create', 'retrieve', 'update' and 'destroy' actions.
    Additionally we also provide an extra 'highlight' action.
    """

    queryset = Lock.objects.all()
    serializer_class = LockSerializer

    def list(self, request):
        queryset = Lock.objects.filter(user=request.user)
        serializer = LockSerializer(queryset, many=True)
        return Response(serializer.data)

    def retrieve(self, request, pk=None):
        queryset = Lock.objects.filter(user=request.user)
        lock = get_object_or_404(queryset, pk=pk)
        serializer = LockSerializer(lock)
        return Response(serializer.data)

    @detail_route(renderer_classes=[renderers.StaticHTMLRenderer])
    def highlight(self, request, *args, **kwargs):
        lock = self.get_object()
        return Response(lock.highlighted)

    def perform_create(self, serializer):
        """Associate user with instance."""
        serializer.save(user=self.request.user)


class EventViewSet(viewsets.ModelViewSet):
    """
    This viewset automatically provides some actions.

    Provides 'list', 'create', 'retrieve', 'update' and 'destroy' actions.
    Additionally we also provide an extra 'highlight' action.
    """

    queryset = Event.objects.all()
    serializer_class = EventSerializer

    @detail_route(renderer_classes=[renderers.StaticHTMLRenderer])
    def highlight(self, request, *args, **kwargs):
        event = self.get_object()
        return Response(event.highlighted)

    def perform_create(self, serializer):
        serializer.save()
