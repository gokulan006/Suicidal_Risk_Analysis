#!/bin/bash
gunicorn app:server --bind 0.0.0.0:$PORT
