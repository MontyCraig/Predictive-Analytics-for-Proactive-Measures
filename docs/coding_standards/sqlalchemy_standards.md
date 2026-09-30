# SQLAlchemy Coding Standards and Best Practices

A comprehensive guide for using SQLAlchemy ORM effectively in Python applications, focusing on database modeling, querying, and performance optimization.

## Table of Contents

1. **Model Design**

    - Base Classes & Mixins

    - Column Types & Relationships

    - Model Configuration

    - Indexes & Constraints

    - Model Validation

2. **Database Operations**

    - Session Management

    - CRUD Operations

    - Query Building

    - Bulk Operations

    - Transaction Handling

3. **Performance**

    - Query Optimization

    - Lazy vs Eager Loading

    - Caching Strategies

    - Connection Pooling

    - Batch Processing

4. **Migration Management**

    - Alembic Integration

    - Migration Scripts

    - Version Control

    - Data Migration

    - Rollback Strategies

5. **Testing & Development**

    - Test Fixtures

    - Database Testing

    - Factory Patterns

    - Development Tools

    - Debugging

---

## 1. Model Design

### Base Classes & Mixins

```python
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, DateTime
from datetime import datetime

Base = declarative_base()

class TimestampMixin:
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

class IDMixin:
    id = Column(Integer, primary_key=True)

class BaseModel(Base, TimestampMixin, IDMixin):
    __abstract__ = True

    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id})>"

```text
### Model Definition

```python
from sqlalchemy import (
    Column, Integer, String, ForeignKey, Table,
    UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSONB

class User(BaseModel):
    __tablename__ = 'users'

    username = Column(String(50), unique=True, nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    settings = Column(JSONB, default=dict, server_default='{}')

    # Relationships

    posts = relationship('Post', back_populates='author')

    # Indexes

    __table_args__ = (
        Index('idx_users_username_email', 'username', 'email'),
    )

class Post(BaseModel):
    __tablename__ = 'posts'

    title = Column(String(200), nullable=False)
    content = Column(Text)
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)

    # Relationships

    author = relationship('User', back_populates='posts')
    tags = relationship('Tag', secondary='post_tags')

    # Indexes

    __table_args__ = (
        Index('idx_posts_author_created', 'author_id', 'created_at'),
    )

```text
---

## 2. Database Operations

### Session Management

```python
from contextlib import contextmanager
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

engine = create_engine(
    'postgresql://user:pass@localhost/dbname',
    pool_size=20,
    max_overflow=0
)
Session = sessionmaker(bind=engine)

@contextmanager
def session_scope():
    """Provide a transactional scope around a series of operations."""
    session = Session()
    try:
        yield session
        session.commit()
    except:
        session.rollback()
        raise
    finally:
        session.close()

# Usage

def create_user(username: str, email: str) -> User:
    with session_scope() as session:
        user = User(username=username, email=email)
        session.add(user)
        session.flush()  # Get the ID before commit

        return user

```text
### Query Building

```python
from sqlalchemy import and_, or_, not_
from typing import List, Optional

class UserRepository:
    def __init__(self, session):
        self.session = session

    def get_by_id(self, user_id: int) -> Optional[User]:
        return self.session.query(User).get(user_id)

    def find_by_username(self, username: str) -> Optional[User]:
        return self.session.query(User).filter(
            User.username == username
        ).first()

    def search_users(
        self,
        keyword: str,
        limit: int = 20,
        offset: int = 0
    ) -> List[User]:
        return self.session.query(User).filter(
            or_(
                User.username.ilike(f"%{keyword}%"),
                User.email.ilike(f"%{keyword}%")
            )
        ).order_by(
            User.created_at.desc()
        ).offset(offset).limit(limit).all()

```text
---

## 3. Performance

### Query Optimization

```python
from sqlalchemy.orm import joinedload, selectinload, contains_eager

class PostRepository:
    def __init__(self, session):
        self.session = session

    def get_posts_with_authors(self, limit: int = 20) -> List[Post]:
        """Efficient loading of posts with their authors."""
        return self.session.query(Post).options(
            joinedload(Post.author)
        ).limit(limit).all()

    def get_user_posts_with_tags(self, user_id: int) -> List[Post]:
        """Efficient loading of posts with tags."""
        return self.session.query(Post).filter(
            Post.author_id == user_id
        ).options(
            selectinload(Post.tags)
        ).all()

    def get_posts_count_by_author(self) -> List[tuple]:
        """Get post counts for each author."""
        return self.session.query(
            User.username,
            func.count(Post.id).label('post_count')
        ).join(
            Post, User.id == Post.author_id
        ).group_by(
            User.username
        ).all()

```text
### Bulk Operations

```python
def bulk_create_users(users_data: List[dict]) -> None:
    """Efficiently insert multiple users."""
    with session_scope() as session:
        session.bulk_insert_mappings(User, users_data)

def bulk_update_posts(posts_updates: List[dict]) -> None:
    """Efficiently update multiple posts."""
    with session_scope() as session:
        session.bulk_update_mappings(Post, posts_updates)

```text
---

## 4. Migration Management

### Alembic Configuration

```python

# alembic/env.py

from alembic import context
from sqlalchemy import engine_from_config, pool

def run_migrations_online():
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix='sqlalchemy.',
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()

```text
### Migration Script

```python

# alembic/versions/123456789_add_user_status.py

"""add user status

Revision ID: 123456789
Revises: previous_revision
Create Date: 2024-01-01 12:00:00.000000
"""

from alembic import op
import sqlalchemy as sa

def upgrade():
    op.add_column('users',
        sa.Column('status',
            sa.String(20),
            nullable=False,
            server_default='active'
        )
    )
    op.create_index(
        'idx_users_status',
        'users',
        ['status']
    )

def downgrade():
    op.drop_index('idx_users_status')
    op.drop_column('users', 'status')

```text
---

## 5. Testing

### Test Fixtures

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@pytest.fixture(scope="session")
def engine():
    return create_engine('postgresql://test:test@localhost/test_db')

@pytest.fixture(scope="session")
def tables(engine):
    BaseModel.metadata.create_all(engine)
    yield
    BaseModel.metadata.drop_all(engine)

@pytest.fixture
def session(engine, tables):
    connection = engine.connect()
    transaction = connection.begin()
    Session = sessionmaker(bind=connection)
    session = Session()

    yield session

    session.close()
    transaction.rollback()
    connection.close()

```text
### Model Tests

```python
def test_user_creation(session):
    user = User(username='test_user', email='test@example.com')
    session.add(user)
    session.commit()

    assert user.id is not None
    assert user.created_at is not None

    fetched_user = session.query(User).get(user.id)
    assert fetched_user.username == 'test_user'

def test_user_posts_relationship(session):
    user = User(username='test_user', email='test@example.com')
    post = Post(title='Test Post', content='Content', author=user)

    session.add_all([user, post])
    session.commit()

    assert post in user.posts
    assert post.author == user

```text
---

## Best Practices

1. **Model Design**

   - Use meaningful table and column names

   - Implement proper relationships and cascades

   - Create appropriate indexes for frequent queries

   - Use appropriate column types and constraints

   - Implement model validation where necessary

2. **Query Optimization**

   - Use appropriate loading strategies (lazy/eager)

   - Minimize the number of database queries

   - Use bulk operations for multiple records

   - Implement proper indexing strategies

   - Monitor and optimize query performance

3. **Session Management**

   - Use session contexts for automatic cleanup

   - Implement proper transaction handling

   - Handle session errors appropriately

   - Use connection pooling in production

   - Close sessions when done

4. **Migration Management**

   - Version control all migrations

   - Test migrations before deployment

   - Implement proper rollback procedures

   - Handle data migrations carefully

   - Document complex migrations

5. **Testing**

   - Use test fixtures for database setup

   - Implement proper test isolation

   - Test all database operations

   - Use factories for test data

   - Test edge cases and error conditions

---

## Conclusion

Following these SQLAlchemy standards ensures:

- Efficient database operations

- Clean and maintainable models

- Optimal query performance

- Reliable data migrations

- Comprehensive testing coverage

Remember to:

- Keep models simple and focused

- Optimize database queries

- Maintain proper documentation

- Implement comprehensive tests

- Monitor database performance

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at <http://www.apache.org/licenses/LICENSE-2.0.>
