pip install --no-cache-dir --force-reinstall numpy==1.23.5
pip install --no-cache-dir --force-reinstall spacy tensorflow
gunicorn -w 4 -b 0.0.0.0:8080 app:flask_app
