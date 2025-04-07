# Pydantic Coding Standards and Best Practices

A comprehensive guide for using Pydantic effectively in Python applications, focusing on data validation, serialization, and type safety.

## Table of Contents

1. **Model Design**
    - Basic Models
    - Field Types & Validation
    - Model Configuration
    - Inheritance & Composition
    - Generic Models

2. **Validation & Type Safety**
    - Field Validators
    - Custom Types
    - Type Annotations
    - Error Handling
    - Complex Validations

3. **Data Conversion**
    - Serialization
    - Deserialization
    - JSON Schema Generation
    - Custom Encoders/Decoders
    - Data Export Formats

4. **Advanced Features**
    - Settings Management
    - Dynamic Model Creation
    - Model Composition
    - Field Aliases
    - Private Attributes

5. **Integration Patterns**
    - FastAPI Integration
    - Database Models
    - Configuration Management
    - API Schemas
    - Testing Strategies

---

## 1. Model Design

### Basic Model Structure

```python
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List
from datetime import datetime

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        from_attributes = True  # For ORM compatibility
        validate_assignment = True  # Validate on attribute assignment
```

### Field Types & Validation

```python
from pydantic import BaseModel, Field, constr, conint, confloat

class Product(BaseModel):
    name: constr(min_length=1, max_length=100)
    price: confloat(gt=0)  # Greater than 0
    stock: conint(ge=0)  # Greater than or equal to 0
    description: Optional[str] = Field(
        None,
        max_length=1000,
        description="Product description"
    )
    tags: List[str] = Field(default_factory=list)
    
    class Config:
        schema_extra = {
            "example": {
                "name": "Sample Product",
                "price": 29.99,
                "stock": 100,
                "description": "A sample product description",
                "tags": ["electronics", "gadgets"]
            }
        }
```

---

## 2. Validation & Type Safety

### Field Validators

```python
from pydantic import BaseModel, validator, root_validator
from typing import List, Dict

class Order(BaseModel):
    items: List[Dict[str, float]]
    total: float
    
    @validator('items')
    def validate_items(cls, v):
        if not v:
            raise ValueError('Order must contain at least one item')
        return v
    
    @root_validator
    def validate_total(cls, values):
        items = values.get('items', [])
        total = values.get('total', 0)
        
        calculated_total = sum(item['price'] for item in items)
        if abs(calculated_total - total) > 0.01:
            raise ValueError(
                f'Total {total} does not match sum of items {calculated_total}'
            )
        return values

class User(BaseModel):
    username: str
    password: str
    password_confirm: str
    
    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.isalnum():
            raise ValueError('Username must be alphanumeric')
        return v
    
    @validator('password_confirm')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
```

### Custom Types

```python
from pydantic import constr, BaseModel
from typing import NewType, Union
from datetime import date

# Custom types with validation
PhoneNumber = constr(regex=r'^\+?1?\d{9,15}$')
PostalCode = constr(regex=r'^\d{5}(-\d{4})?$')
Age = conint(ge=0, le=150)

# Custom model with type validation
class Address(BaseModel):
    street: str
    city: str
    postal_code: PostalCode
    country: str

class Contact(BaseModel):
    phone: PhoneNumber
    email: EmailStr
    address: Address
```

---

## 3. Data Conversion

### Serialization & Deserialization

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Dict, Any

class User(BaseModel):
    id: int
    username: str
    created_at: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
    
    def dict_with_formatted_datetime(self) -> Dict[str, Any]:
        data = self.model_dump()
        data['created_at'] = self.created_at.strftime('%Y-%m-%d %H:%M:%S')
        return data

# Usage
user = User(id=1, username="john_doe", created_at=datetime.utcnow())
json_data = user.model_dump_json()  # For JSON serialization
dict_data = user.model_dump()  # For dict conversion
formatted_data = user.dict_with_formatted_datetime()
```

### Custom Encoders

```python
from pydantic import BaseModel
from datetime import datetime, date
from decimal import Decimal
from typing import Any

class CustomJSONEncoder(BaseModel):
    class Config:
        json_encoders = {
            datetime: lambda v: v.strftime("%Y-%m-%d %H:%M:%S"),
            date: lambda v: v.strftime("%Y-%m-%d"),
            Decimal: lambda v: float(v),
            bytes: lambda v: v.decode(),
        }

class Transaction(CustomJSONEncoder):
    id: int
    amount: Decimal
    timestamp: datetime
    data: bytes
```

---

## 4. Advanced Features

### Settings Management

```python
from pydantic_settings import BaseSettings
from typing import Optional, Dict
from functools import lru_cache

class Settings(BaseSettings):
    APP_NAME: str = "MyApp"
    DEBUG: bool = False
    DATABASE_URL: str
    REDIS_URL: Optional[str] = None
    API_KEYS: Dict[str, str] = {}
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

### Dynamic Model Creation

```python
from pydantic import create_model, BaseModel
from typing import Dict, Any

def create_dynamic_model(
    name: str,
    fields: Dict[str, Any]
) -> type[BaseModel]:
    """Create a Pydantic model dynamically."""
    return create_model(
        name,
        **{
            field_name: (field_type, ...)
            for field_name, field_type in fields.items()
        }
    )

# Usage
UserModel = create_dynamic_model(
    'User',
    {
        'username': str,
        'email': str,
        'age': int
    }
)
```

---

## 5. Integration Patterns

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr, ValidationError

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: int
    username: str
    email: EmailStr

app = FastAPI()

@app.post("/users/", response_model=UserResponse)
async def create_user(user: UserCreate):
    try:
        # Pydantic handles validation automatically
        user_dict = user.model_dump()
        # Process user creation...
        return UserResponse(id=1, **user_dict)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
```

### Database Integration

```python
from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel

SQLAlchemyBase = declarative_base()

class UserDB(SQLAlchemyBase):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True)
    email = Column(String, unique=True)

class UserSchema(BaseModel):
    id: Optional[int] = None
    username: str
    email: EmailStr
    
    class Config:
        from_attributes = True

# Usage
def get_user(db: Session, user_id: int) -> UserSchema:
    db_user = db.query(UserDB).filter(UserDB.id == user_id).first()
    return UserSchema.from_orm(db_user)
```

---

## Testing

### Model Testing

```python
import pytest
from pydantic import ValidationError

def test_user_validation():
    # Valid user
    user = User(username="john_doe", email="john@example.com")
    assert user.username == "john_doe"
    
    # Invalid email
    with pytest.raises(ValidationError) as exc_info:
        User(username="john_doe", email="invalid-email")
    assert "value is not a valid email address" in str(exc_info.value)

def test_product_price_validation():
    # Invalid price
    with pytest.raises(ValidationError) as exc_info:
        Product(name="Test", price=-10)
    assert "ensure this value is greater than 0" in str(exc_info.value)
```

---

## Best Practices

1. **Model Design**
   - Keep models focused and single-purpose
   - Use inheritance to share common fields
   - Leverage type hints for better IDE support
   - Document models with clear field descriptions

2. **Validation**
   - Use built-in validators when possible
   - Create reusable custom validators
   - Handle validation errors gracefully
   - Validate at the edge of your application

3. **Performance**
   - Cache compiled models when possible
   - Use `Config.arbitrary_types_allowed = True` only when necessary
   - Be mindful of recursive models
   - Profile validation performance for large datasets

4. **Security**
   - Never store sensitive data in plain text
   - Use secure field types for passwords
   - Implement proper data sanitization
   - Validate input at system boundaries

5. **Maintenance**
   - Keep models up to date with schema changes
   - Document complex validation logic
   - Use consistent naming conventions
   - Regular security updates for dependencies

---

## Conclusion

Following these Pydantic standards ensures:
- Type-safe data validation
- Clean and maintainable data models
- Efficient serialization/deserialization
- Secure data handling
- Excellent IDE support

Remember to:
- Keep up with Pydantic updates
- Follow type hinting best practices
- Maintain comprehensive tests
- Document complex validations
- Profile performance for critical paths

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at http://www.apache.org/licenses/LICENSE-2.0.
