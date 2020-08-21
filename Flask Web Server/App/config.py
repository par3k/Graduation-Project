class Config(object):
    SECRET_KEY = 'your secret key'
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql:/your sql addr'
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CSRF_ENABLED = True
    DEBUG = True

