#!/usr/local/miniconda3/envs/medea python
# -*- coding: utf-8 -*-

import datetime
import os
import json
from datetime import date
from pathlib import Path

def set_log_file(log_path):

    """Imposta il percorso del file di log globale."""
    if os.path.isfile(log_path): os.remove(log_path)
    global LOGFILE

    LOGFILE = log_path


def log_message(message):
    """Scrive un messaggio nel file di log con timestamp."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    with open(LOGFILE, "a") as log:
        log.write(log_entry)
    print(log_entry.strip())



if __name__ == "__main__":
    set_log_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logfile.log'))
    log_message('Prova logfile')