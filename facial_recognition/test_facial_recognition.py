"""Test file for facial_recognition.py."""


import os


def test_recognizer_generates_file():
    """Test that the  trainer finishes and the "brain file" is generated."""
    from facial_recognition import train_recognizer
    if os.isfile('recog_brain2.yml') is True:
        os.remove('recog_brain2.yml')
    train_recognizer(save_file='recog_brain2.yml')
    assert os.isfile('recog_brain2.yml') is True
    os.remove('recog_brain2.yml')
