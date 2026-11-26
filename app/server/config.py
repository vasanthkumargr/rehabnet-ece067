import os

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(APP_DIR)
DATA_DIR = os.environ.get("REHABNET_DATA_DIR", os.path.join(APP_DIR, "data"))
WEB_DIR = os.path.join(APP_DIR, "web")
REF_DIR = os.path.join(APP_DIR, "data", "references")
DB_URL = os.environ.get("REHABNET_DB_URL", "sqlite:///" + os.path.join(DATA_DIR, "rehabnet.db").replace("\\", "/"))
