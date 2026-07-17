#!/usr/bin/python
# -*- encoding: utf-8 -*-

import re
import unicodedata
from builtins import str as unicode
from jttts.tts_model.frontend.en_normalization.numbers import normalize_numbers

import pdb

def normalize(sentence):
    """ Normalize English text.
    """
    # preprocessing
    sentence = unicode(sentence)
    sentence = normalize_numbers(sentence)
    sentence = ''.join(
        char for char in unicodedata.normalize('NFD', sentence)
        if unicodedata.category(char) != 'Mn')  # Strip accents
    sentence = sentence.lower()
    sentence = re.sub(r"[^ a-z'.,?!\-]", "", sentence)
    sentence = sentence.replace("i.e.", "that is")
    sentence = sentence.replace("e.g.", "for example")
    return sentence
