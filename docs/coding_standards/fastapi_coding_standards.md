# FastAPI Coding Standards and Best Practices

A comprehensive guide for building high-performance, type-safe APIs using FastAPI following enterprise-level best practices.

## Table of Contents

1. **Project Structure**
    - Directory Layout
    - Application Organization
    - Configuration Management
    - Dependency Management

2. **API Design**
    - Route Organization
    - Path Operations
    - Request/Response Models
    - Dependencies & Middleware
    - Error Handling

3. **Data Validation**
    - Pydantic Models
    - Request Validation
    - Response Schemas
    - Custom Validators
    - Type Annotations

4. **Security**
    - Authentication
    - Authorization
    - OAuth2 & JWT
    - CORS & Security Headers
    - Rate Limiting

5. **Performance**
    - Async Operations
    - Background Tasks
    - Caching
    - Database Access
    - Profiling & Monitoring

---

## 1. Project Structure

### Directory Layout
```
fastapi_project/
├── alembic/
│   └── versions/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   └── dependencies/
│   │   └── deps.py
│   ├── core/
│   │   ├── config.py
│   │   └── security.py
│   ├── db/
│   │   ├── base.py
│   │   └── session.py
│   ├── models/
│   ├── schemas/
│   └── services/
├── tests/
├── .env
├── main.py
└── requirements.txt
```

### Application Configuration
```python
# app/core/config.py
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "FastAPI Project"
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
```

### Dependencies Management
```python
# app/api/deps.py
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import Generator

from app.db.session import SessionLocal
from app.core import security

def get_db() -> Generator:
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()

async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(security.oauth2_scheme)
) -> models.User:
    user = await security.get_current_user(db, token)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )
    return user
```

---

## 2. API Design

### Route Organization
```python
# app/api/v1/api.py
from fastapi import APIRouter
from app.api.v1.endpoints import users, items, auth

api_router = APIRouter()

api_router.include_router(auth.router, tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(items.router, prefix="/items", tags=["items"])
```

### Path Operations
```python
# app/api/v1/endpoints/users.py
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from app import schemas, models, crud
from app.api import deps

router = APIRouter()

@router.get("/", response_model=List[schemas.User])
async def read_users(
    db: Session = Depends(deps.get_db),
    skip: int = 0,
    limit: int = 100,
    current_user: models.User = Depends(deps.get_current_user),
):
    """
    Retrieve users.
    """
    users = await crud.user.get_multi(db, skip=skip, limit=limit)
    return users

@router.post("/", response_model=schemas.User)
async def create_user(
    *,
    db: Session = Depends(deps.get_db),
    user_in: schemas.UserCreate,
):
    """
    Create new user.
    """
    user = await crud.user.get_by_email(db, email=user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists."
        )
    user = await crud.user.create(db, obj_in=user_in)
    return user
```

---

## 3. Data Validation

### Pydantic Models
```python
# app/schemas/user.py
from pydantic import BaseModel, EmailStr, constr
from typing import Optional

class UserBase(BaseModel):
    email: Optional[EmailStr] = None
    is_active: Optional[bool] = True
    full_name: Optional[str] = None

class UserCreate(UserBase):
    email: EmailStr
    password: constr(min_length=8)

class UserUpdate(UserBase):
    password: Optional[constr(min_length=8)] = None

class UserInDBBase(UserBase):
    id: int
    
    class Config:
        from_attributes = True

class User(UserInDBBase):
    pass

class UserInDB(UserInDBBase):
    hashed_password: str
```

### Custom Validators
```python
# app/schemas/validators.py
from pydantic import validator, EmailStr
from datetime import datetime
from typing import Optional

class ItemCreate(BaseModel):
    title: str
    description: Optional[str] = None
    price: float
    tax: Optional[float] = None
    
    @validator('price')
    def price_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError('Price must be positive')
        return v
    
    @validator('tax')
    def tax_must_be_valid(cls, v, values):
        if v is not None:
            if v < 0:
                raise ValueError('Tax must not be negative')
            if v > values['price']:
                raise ValueError('Tax cannot be greater than price')
        return v
```

---

## 4. Security

### Authentication Setup
```python
# app/core/security.py
from datetime import datetime, timedelta
from typing import Any, Union
from jose import jwt
from passlib.context import CryptContext
from fastapi.security import OAuth2PasswordBearer

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(
    subject: Union[str, Any], expires_delta: timedelta = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(
        to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM
    )
    return encoded_jwt

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)
```

### Middleware Configuration
```python
# main.py
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from app.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# Set all CORS enabled origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 5. Performance

### Async Database Access
```python
# app/db/session.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

engine = create_async_engine(
    settings.SQLALCHEMY_DATABASE_URI,
    pool_pre_ping=True,
    echo=settings.DB_ECHO
)

AsyncSessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

async def get_async_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
```

### Background Tasks
```python
# app/api/v1/endpoints/items.py
from fastapi import BackgroundTasks
from app.services.notifications import send_notification

@router.post("/items/")
async def create_item(
    item: schemas.ItemCreate,
    background_tasks: BackgroundTasks,
    current_user: models.User = Depends(deps.get_current_user),
):
    """
    Create new item with background notification.
    """
    item = await crud.item.create(item=item, user_id=current_user.id)
    background_tasks.add_task(
        send_notification,
        email=current_user.email,
        message=f"Item {item.id} has been created"
    )
    return item
```

### Caching
```python
# app/core/cache.py
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache

@router.get("/items/", response_model=List[schemas.Item])
@cache(expire=300)  # Cache for 5 minutes
async def read_items(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(deps.get_db)
):
    """
    Retrieve items with caching.
    """
    items = await crud.item.get_multi(db, skip=skip, limit=limit)
    return items
```

---

## Testing

### Test Setup
```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.main import app
from app.db.session import SessionLocal
from app.core.config import settings

@pytest.fixture(scope="session")
def db() -> Generator:
    yield SessionLocal()

@pytest.fixture(scope="module")
def client() -> Generator:
    with TestClient(app) as c:
        yield c

@pytest.fixture(scope="module")
def superuser_token_headers(client: TestClient) -> Dict[str, str]:
    return get_superuser_token_headers(client)
```

### API Tests
```python
# tests/api/v1/test_users.py
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

def test_create_user(
    client: TestClient, superuser_token_headers: dict, db: Session
) -> None:
    data = {
        "email": "test@example.com",
        "password": "test123456",
        "full_name": "Test User"
    }
    response = client.post(
        f"{settings.API_V1_STR}/users/",
        headers=superuser_token_headers,
        json=data,
    )
    assert response.status_code == 200
    content = response.json()
    assert content["email"] == data["email"]
    assert content["full_name"] == data["full_name"]
```

---

## Documentation

### API Documentation
```python
# main.py
from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi

app = FastAPI(
    title="My API",
    description="This is a very fancy project",
    version="2.5.0",
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Custom title",
        version="2.5.0",
        description="This is a very custom OpenAPI schema",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi
```

---

## Conclusion

Following these FastAPI coding standards ensures:
- Type-safe and maintainable APIs
- High-performance async operations
- Secure authentication and authorization
- Clean and organized codebase
- Comprehensive testing coverage
- Clear API documentation

Remember to:
- Keep dependencies up to date
- Monitor performance metrics
- Regularly review security practices
- Update documentation as the API evolves
- Follow FastAPI's latest best practices and updates

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at http://www.apache.org/licenses/LICENSE-2.0.
``` 


## License

Copyright © 2024-2025 MetaReps, Inc. All rights reserved.

This is proprietary software. Unauthorized copying, modification, distribution, 
or use of this software, via any medium, is strictly prohibited without the 
express written permission of MetaReps, Inc.
