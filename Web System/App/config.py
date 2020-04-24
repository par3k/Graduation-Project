class Config(object):
    SECRET_KEY = '608f10b6e0f01d77436fb7226bc46b86'
    SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://root:ghlwo7831@localhost/graduation'
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CSRF_ENABLED = True
    DEBUG = True

