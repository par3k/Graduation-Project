from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager
# from .config import Config
import pymysql
pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your scret key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://your sql addr'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'
login_manager.login_message_category = 'info'


def create_app():
    # app.config.from_object(Config)
    app.debug = True
    db.init_app(app)
    login_manager.init_app(app)
    return app


from App import routes
