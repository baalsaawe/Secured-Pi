from django.views.generic import UpdateView, DeleteView
from securedpi_locks.models import Lock
from securedpi_events.models import Event
import requests
from django.http import HttpResponseRedirect
from django.urls import reverse, reverse_lazy
from securedpi.settings import FLASK_SERVER


class EditLockView(UpdateView):
    """Define edit Lock class."""

    template_name = 'securedpi_locks/edit_lock.html'
    model = Lock
    fields = [
        'name',
        'location',
        'description',
        'is_active',
        'facial_recognition'
    ]

    def get_success_url(self):
        """Set redirection after updating the album."""
        url = reverse('dashboard')
        return url


class DeleteLockView(DeleteView):
    """Delete class to delete a lock."""

    model = Lock
    success_url = reverse_lazy('dashboard')


def manual_action(request, **kwargs):
    """
    Create an instance of Event.

    Make a request to raspberry pi to lock/unlock.
    Chanage lock status to pending. Redirect to dashboard
    """
    lock = Lock.objects.get(pk=kwargs['pk'])
    data = {
        'lock_id': kwargs['pk'],
        'action': kwargs['action'],
        'serial': lock.serial,
        'mtype': 'manual'
    }
    new_event = Event(**data)
    new_event.save()
    data['event_id'] = new_event.pk
    lock.status = 'pending'
    lock.save()
    requests.post(FLASK_SERVER + ':5000', json=data)
    return HttpResponseRedirect(reverse('dashboard'))
