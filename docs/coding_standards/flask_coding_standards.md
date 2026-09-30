# Flask Coding Standards and Best Practices

A comprehensive guide for building scalable, maintainable web applications using Flask following enterprise-level best practices.

## Table of Contents

1. **Project Structure**

    - Application Factory Pattern

    - Blueprints Organization

    - Configuration Management

    - Extensions Setup

2. **Application Design**

    - Route Organization

    - View Functions

    - Templates & Static Files

    - Forms & Validation

    - Database Integration

3. **Security**

    - Authentication & Authorization

    - CSRF Protection

    - Session Management

    - Password Hashing

    - Security Headers

4. **Performance**

    - Caching Strategies

    - Database Optimization

    - Asset Management

    - Request Processing

    - Background Tasks

5. **Testing & Quality**

    - Unit Testing

    - Integration Testing

    - Test Coverage

    - Code Quality

    - Documentation

---

## 1. Project Structure

### Application Factory Pattern

```python

# app/__init__.py

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from config import config

db = SQLAlchemy()
login_manager = LoginManager()

def create_app(config_name):
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    config[config_name].init_app(app)

    # Initialize extensions

    db.init_app(app)
    login_manager.init_app(app)

    # Register blueprints

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)

    from .auth import auth as auth_blueprint
    app.register_blueprint(auth_blueprint, url_prefix='/auth')

    return app

```text
### Project Layout

```text
flask_project/
├── app/
│   ├── __init__.py
│   ├── models/
│   ├── views/
│   ├── templates/
│   ├── static/
│   └── utils/
├── tests/
├── config.py
├── requirements/
│   ├── base.txt
│   ├── dev.txt
│   └── prod.txt
├── migrations/
└── wsgi.py

```text
### Configuration Management

```python

# config.py

import os
from dotenv import load_dotenv

basedir = os.path.abspath(os.path.dirname(__file__))
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'hard-to-guess-string'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    @staticmethod
    def init_app(app):
        pass

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL')

class ProductionConfig(Config):
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

    @classmethod
    def init_app(cls, app):
        Config.init_app(app)

        # Production-specific logging

        import logging
        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            'logs/flask_app.log',
            maxBytes=10240,
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

```text
---

## 2. Application Design

### Blueprint Organization

```python

# app/main/__init__.py

from flask import Blueprint

main = Blueprint('main', __name__)

from . import views, errors

# app/main/views.py

from flask import render_template
from . import main
from ..models import User

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/user/<username>')
def user(username):
    user = User.query.filter_by(username=username).first_or_404()
    return render_template('user.html', user=user)

```text
### Forms & Validation

```python

# app/auth/forms.py

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Email, Length

class LoginForm(FlaskForm):
    email = StringField('Email', validators=[
        DataRequired(),
        Email()
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8)
    ])
    submit = SubmitField('Log In')

# app/auth/views.py

@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.verify_password(form.password.data):
            login_user(user)
            return redirect(url_for('main.index'))
        flash('Invalid email or password.')
    return render_template('auth/login.html', form=form)

```text
---

## 3. Security

### Authentication Setup

```python

# app/models.py

from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from . import db, login_manager

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(64), unique=True, index=True)
    username = db.Column(db.String(64), unique=True, index=True)
    password_hash = db.Column(db.String(128))

    @property
    def password(self):
        raise AttributeError('password is not a readable attribute')

    @password.setter
    def password(self, password):
        self.password_hash = generate_password_hash(password)

    def verify_password(self, password):
        return check_password_hash(self.password_hash, password)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

```text
### Security Headers

```python

# app/__init__.py

from flask_talisman import Talisman

def create_app(config_name):
    app = Flask(__name__)
    Talisman(app, content_security_policy={
        'default-src': "'self'",
        'img-src': '*',
        'script-src': ["'self'", "'unsafe-inline'"],
        'style-src': ["'self'", "'unsafe-inline'"]
    })

```text
---

## 4. Performance

### Caching Strategy

```python

# app/extensions.py

from flask_caching import Cache

cache = Cache()

# app/__init__.py

def create_app(config_name):
    app = Flask(__name__)
    cache.init_app(app, config={
        'CACHE_TYPE': 'redis',
        'CACHE_REDIS_URL': os.environ.get('REDIS_URL')
    })

# app/views.py

@main.route('/users')
@cache.cached(timeout=300)  # Cache for 5 minutes

def get_users():
    users = User.query.all()
    return render_template('users.html', users=users)

```text
### Database Optimization

```python

# app/models.py

class Post(db.Model):
    __tablename__ = 'posts'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255), index=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'))

    __table_args__ = (
        db.Index('idx_user_created', user_id, 'created_at'),
    )

    @classmethod
    def get_user_posts(cls, user_id):
        return cls.query.options(
            db.joinedload('user')
        ).filter_by(user_id=user_id).all()

```text
---

## 5. Testing

### Test Configuration

```python

# tests/conftest.py

import pytest
from app import create_app, db
from app.models import User

@pytest.fixture
def app():
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.session.remove()
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def runner(app):
    return app.test_cli_runner()

```text
### Unit Tests

```python

# tests/test_models.py

def test_password_setter():
    u = User(password='cat')
    assert u.password_hash is not None

def test_no_password_getter():
    u = User(password='cat')
    with pytest.raises(AttributeError):
        u.password

def test_password_verification():
    u = User(password='cat')
    assert u.verify_password('cat')
    assert not u.verify_password('dog')

```text
---

## CLI Commands

### Custom Commands

```python

# app/commands.py

import click
from flask.cli import with_appcontext
from . import db

@click.command('init-db')
@with_appcontext
def init_db_command():
    """Clear existing data and create new tables."""
    db.drop_all()
    db.create_all()
    click.echo('Initialized the database.')

def init_app(app):
    app.cli.add_command(init_db_command)

```text
---

## Error Handling

### Custom Error Pages

```python

# app/errors.py

from flask import render_template
from . import main

@main.app_errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@main.app_errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

```text
---

## Documentation

### API Documentation

```python
from flask_restx import Api, Resource, fields

api = Api(
    title='Flask API',
    version='1.0',
    description='A simple Flask API'
)

user_model = api.model('User', {
    'id': fields.Integer,
    'username': fields.String,
    'email': fields.String
})

@api.route('/users')
class UserList(Resource):
    @api.doc('list_users')
    @api.marshal_list_with(user_model)
    def get(self):
        """List all users"""
        return User.query.all()

```text
---

## Conclusion

Following these Flask coding standards ensures:

- Clean and maintainable code structure

- Secure application design

- Optimized performance

- Comprehensive testing coverage

- Clear documentation

Remember to:

- Keep dependencies updated

- Follow Flask's latest best practices

- Regularly review security measures

- Maintain comprehensive documentation

- Monitor application performance

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at <http://www.apache.org/licenses/LICENSE-2.0.>

```text
