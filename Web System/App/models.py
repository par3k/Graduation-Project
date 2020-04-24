# -*- coding: utf-8 -*-
from . import db, login_manager
from flask_login import UserMixin
from datetime import datetime

# 데이터 베이스

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(user_id)


class User(db.Model, UserMixin):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)  # For @login_manager.user_loader
    name = db.Column(db.CHAR(20), unique=True, nullable=False)
    email = db.Column(db.CHAR(120), unique=True, nullable=False)
    password = db.Column(db.VARCHAR(100), nullable=False)

    posts = db.relationship('Post', backref='author', lazy=True)  # For Post the Notice

    def __repr__(self):
        return f"Patient('{self.name}','{self.email}')"


class Post(db.Model):  # For Notice
    __tablename__ = 'post'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100), nullable=False)
    content = db.Column(db.Text, nullable=False)
    date_posted = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    category = db.Column(db.String(40), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

    def __repr__(self):
        return f"Post('{self.title}', '{self.date_posted}')"

