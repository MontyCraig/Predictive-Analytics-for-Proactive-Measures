# Python API Client Standards and Best Practices

A comprehensive guide for building robust, maintainable, and user-friendly API clients in Python.

## Table of Contents

1. **Client Architecture**

    - Client Structure

    - Resource Organization

    - Configuration Management

    - Authentication Handling

    - Error Handling

2. **Request Handling**

    - HTTP Methods

    - Request Building

    - Parameter Validation

    - Request Middleware

    - Retry Logic

3. **Response Handling**

    - Response Parsing

    - Data Models

    - Error Responses

    - Rate Limiting

    - Pagination

4. **Authentication**

    - Authentication Methods

    - Token Management

    - Refresh Logic

    - Security Best Practices

    - Multi-tenant Support

5. **Advanced Features**

    - Async Support

    - Caching

    - Logging

    - Testing

    - Documentation

---

## 1. Client Architecture

### Basic Client Structure

```python
from typing import Optional, Dict, Any
import httpx
from pydantic import BaseModel, Field

class APIConfig(BaseModel):
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    verify_ssl: bool = True

class APIClient:
    def __init__(self, config: APIConfig):
        self.config = config
        self.client = httpx.Client(
            base_url=config.base_url,
            timeout=config.timeout,
            verify=config.verify_ssl
        )
        self._setup_auth()

    def _setup_auth(self):
        if self.config.api_key:
            self.client.headers.update({
                "Authorization": f"Bearer {self.config.api_key}"
            })

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

```text
### Resource Organization

```python
from abc import ABC, abstractmethod

class BaseResource(ABC):
    def __init__(self, client: APIClient):
        self.client = client

    @abstractmethod
    def base_path(self) -> str:
        pass

class UsersResource(BaseResource):
    def base_path(self) -> str:
        return "/users"

    def list(self, page: int = 1, limit: int = 20):
        return self.client.get(
            f"{self.base_path()}",
            params={"page": page, "limit": limit}
        )

    def get(self, user_id: int):
        return self.client.get(f"{self.base_path()}/{user_id}")

    def create(self, data: Dict[str, Any]):
        return self.client.post(self.base_path(), json=data)

class Client(APIClient):
    def __init__(self, config: APIConfig):
        super().__init__(config)
        self.users = UsersResource(self)

```text
---

## 2. Request Handling

### Request Building

```python
from typing import Optional, Dict, Any, Union
from urllib.parse import urljoin

class RequestBuilder:
    def __init__(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.method = method.upper()
        self.url = url
        self.params = params or {}
        self.data = data
        self.json = json
        self.headers = headers or {}

    def add_param(self, key: str, value: Any) -> 'RequestBuilder':
        if value is not None:
            self.params[key] = value
        return self

    def add_header(self, key: str, value: str) -> 'RequestBuilder':
        self.headers[key] = value
        return self

    def build(self) -> Dict[str, Any]:
        request = {
            "method": self.method,
            "url": self.url,
            "params": self.params,
            "headers": self.headers
        }

        if self.json is not None:
            request["json"] = self.json
        elif self.data is not None:
            request["data"] = self.data

        return request

```text
### Retry Logic

```python
import time
from typing import List, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

class RetryConfig:
    def __init__(
        self,
        max_attempts: int = 3,
        max_wait: int = 60,
        retry_codes: Optional[List[int]] = None
    ):
        self.max_attempts = max_attempts
        self.max_wait = max_wait
        self.retry_codes = retry_codes or [429, 500, 502, 503, 504]

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (httpx.TimeoutException, httpx.HTTPError)
    )
)
def make_request(client: httpx.Client, **kwargs):
    response = client.request(**kwargs)
    response.raise_for_status()
    return response

```text
---

## 3. Response Handling

### Response Models

```python
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

class User(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime
    updated_at: datetime
    is_active: bool = True

class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    page: int
    pages: int
    has_next: bool
    has_prev: bool

class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class APIResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[ErrorResponse] = None
    meta: Optional[Dict[str, Any]] = None

```text
### Response Processing

```python
from typing import TypeVar, Type, Generic

T = TypeVar('T', bound=BaseModel)

class ResponseProcessor(Generic[T]):
    def __init__(self, model: Type[T]):
        self.model = model

    def process(self, response: httpx.Response) -> T:
        data = response.json()

        if not response.is_success:
            error = ErrorResponse(**data)
            raise APIError(error)

        return self.model(**data)

class UserProcessor(ResponseProcessor[User]):
    def process_list(
        self,
        response: httpx.Response
    ) -> PaginatedResponse:
        data = response.json()
        items = [User(**item) for item in data["items"]]
        return PaginatedResponse(items=items, **data["meta"])

```text
---

## 4. Authentication

### Authentication Handler

```python
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

class AuthHandler(ABC):
    @abstractmethod
    def get_auth_header(self) -> Dict[str, str]:
        pass

    @abstractmethod
    def refresh_auth(self) -> None:
        pass

class BearerAuth(AuthHandler):
    def __init__(
        self,
        token: str,
        refresh_token: Optional[str] = None,
        expires_in: Optional[int] = None
    ):
        self.token = token
        self.refresh_token = refresh_token
        self.expires_at = (
            datetime.utcnow() + timedelta(seconds=expires_in)
            if expires_in else None
        )

    def get_auth_header(self) -> Dict[str, str]:
        if self.should_refresh():
            self.refresh_auth()
        return {"Authorization": f"Bearer {self.token}"}

    def should_refresh(self) -> bool:
        if not self.expires_at:
            return False
        return datetime.utcnow() >= self.expires_at - timedelta(minutes=5)


    def refresh_auth(self) -> None:
        if not self.refresh_token:
            raise ValueError("No refresh token available")
        # Implement token refresh logic here

```text
---

## 5. Advanced Features

### Async Support

```python
import asyncio
from typing import AsyncIterator
import httpx

class AsyncAPIClient:
    def __init__(self, config: APIConfig):
        self.config = config
        self.client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=config.timeout,
            verify=config.verify_ssl
        )

    async def __aenter__(self) -> 'AsyncAPIClient':
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        await self.client.aclose()

    async def get(self, url: str, **kwargs):
        async with self.client as client:
            response = await client.get(url, **kwargs)
            return response.json()

    async def paginate(
        self,
        url: str,
        params: Dict[str, Any]
    ) -> AsyncIterator[Dict[str, Any]]:
        page = 1
        while True:
            params["page"] = page
            response = await self.get(url, params=params)
            yield response

            if not response["meta"]["has_next"]:
                break
            page += 1

```text
### Caching

```python
from functools import wraps
from typing import Optional, Callable
import hashlib
import json

class Cache:
    def __init__(self, ttl: int = 300):
        self.ttl = ttl
        self.cache = {}

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.utcnow().timestamp() - timestamp < self.ttl:

                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: Any):
        self.cache[key] = (value, datetime.utcnow().timestamp())

def cached(ttl: int = 300):
    def decorator(func: Callable):
        cache = Cache(ttl=ttl)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function args

            key_parts = [
                func.__name__,
                str(args),
                str(sorted(kwargs.items()))
            ]
            key = hashlib.md5(
                json.dumps(key_parts).encode()
            ).hexdigest()

            # Check cache

            cached_value = cache.get(key)
            if cached_value is not None:
                return cached_value

            # Execute function and cache result

            result = func(*args, **kwargs)
            cache.set(key, result)
            return result

        return wrapper
    return decorator

```text
---

## Best Practices

1. **Client Design**

   - Use resource-based organization

   - Implement proper error handling

   - Support configuration management

   - Follow HTTP standards

   - Provide clear documentation

2. **Authentication**

   - Secure credential handling

   - Implement token refresh

   - Support multiple auth methods

   - Handle multi-tenant scenarios

   - Follow security best practices

3. **Performance**

   - Implement connection pooling

   - Use appropriate timeouts

   - Implement caching where appropriate

   - Support async operations

   - Handle rate limiting

4. **Error Handling**

   - Use custom exceptions

   - Provide detailed error messages

   - Implement retry logic

   - Handle network errors

   - Validate responses

5. **Testing**

   - Mock HTTP responses

   - Test error scenarios

   - Verify retry logic

   - Test async operations

   - Use integration tests

---

## Conclusion

Following these API client standards ensures:

- Reliable API interactions

- Maintainable client code

- Efficient resource usage

- Proper error handling

- Clear documentation

Remember to:

- Follow HTTP best practices

- Handle errors gracefully

- Implement proper testing

- Document client usage

- Monitor client performance

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at <http://www.apache.org/licenses/LICENSE-2.0.>
