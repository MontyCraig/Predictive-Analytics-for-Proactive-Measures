# Python Microservices Standards and Best Practices

A comprehensive guide for building, deploying, and maintaining microservices using Python, focusing on scalability, reliability, and maintainability.

## Table of Contents

1. **Architecture**
    - Service Design
    - API Design
    - Data Management
    - Service Communication
    - Event Handling

2. **Implementation**
    - Project Structure
    - Code Organization
    - Configuration
    - Dependency Management
    - Testing Strategy

3. **Infrastructure**
    - Containerization
    - Orchestration
    - Service Discovery
    - Load Balancing
    - Monitoring

4. **Resilience**
    - Circuit Breaking
    - Rate Limiting
    - Retries & Timeouts
    - Health Checks
    - Fault Tolerance

5. **Operations**
    - Deployment
    - Scaling
    - Monitoring
    - Logging
    - Security

---

## 1. Architecture

### Service Design
```python
from abc import ABC, abstractmethod
from typing import Dict, Any
from pydantic import BaseModel

class ServiceBase(ABC):
    """Base class for microservice implementation."""
    
    @abstractmethod
    async def start(self):
        """Initialize service resources."""
        pass
    
    @abstractmethod
    async def stop(self):
        """Cleanup service resources."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        pass

class ServiceConfig(BaseModel):
    """Service configuration model."""
    service_name: str
    version: str
    port: int
    dependencies: Dict[str, str]
    environment: str = "development"

class MicroService(ServiceBase):
    def __init__(self, config: ServiceConfig):
        self.config = config
        self.is_healthy = False
        self.dependencies_status = {}
    
    async def start(self):
        """Start service and initialize resources."""
        # Initialize resources
        await self._init_database()
        await self._init_cache()
        await self._init_message_queue()
        
        # Check dependencies
        await self._check_dependencies()
        
        self.is_healthy = True
    
    async def stop(self):
        """Gracefully shutdown service."""
        # Cleanup resources
        await self._cleanup_database()
        await self._cleanup_cache()
        await self._cleanup_message_queue()
        
        self.is_healthy = False
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "service": self.config.service_name,
            "version": self.config.version,
            "status": "healthy" if self.is_healthy else "unhealthy",
            "dependencies": self.dependencies_status
        }
```

### API Design
```python
from fastapi import FastAPI, HTTPException, Depends
from typing import Optional, List
from pydantic import BaseModel, Field

# API Models
class ErrorResponse(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None

class SuccessResponse(BaseModel):
    data: Any
    meta: Optional[Dict[str, Any]] = None

# API Versioning
class APIVersion:
    def __init__(self, version: int):
        self.version = version
    
    def __call__(self):
        return self.version

# API Setup
app = FastAPI(
    title="User Service",
    description="User management microservice",
    version="1.0.0"
)

# Middleware for version handling
@app.middleware("http")
async def version_middleware(request, call_next):
    version = request.headers.get("API-Version", "1")
    request.state.version = int(version)
    response = await call_next(request)
    return response

# API Routes
@app.get(
    "/users/{user_id}",
    response_model=SuccessResponse,
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def get_user(
    user_id: int,
    version: int = Depends(APIVersion(1))
):
    try:
        # Version-specific logic
        if version == 1:
            user = await user_service.get_user_v1(user_id)
        else:
            user = await user_service.get_user_v2(user_id)
        
        return SuccessResponse(data=user)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
```

---

## 2. Implementation

### Project Structure
```
microservice/
├── src/
│   └── service_name/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── routes.py
│       │   └── models.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   └── service.py
│       ├── db/
│       │   ├── __init__.py
│       │   └── models.py
│       └── utils/
│           ├── __init__.py
│           └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   └── test_api.py
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

### Service Implementation
```python
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from redis import Redis
from kafka import KafkaProducer
import aiohttp

class ServiceDependencies:
    """Service dependencies container."""
    
    def __init__(
        self,
        db: AsyncSession,
        cache: Redis,
        producer: KafkaProducer,
        http_client: aiohttp.ClientSession
    ):
        self.db = db
        self.cache = cache
        self.producer = producer
        self.http_client = http_client

class UserService:
    """Example microservice implementation."""
    
    def __init__(self, deps: ServiceDependencies):
        self.deps = deps
    
    async def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user with caching."""
        # Try cache first
        cached_user = await self.deps.cache.get(f"user:{user_id}")
        if cached_user:
            return cached_user
        
        # Query database
        user = await self.deps.db.query(User).get(user_id)
        if user:
            # Cache result
            await self.deps.cache.set(
                f"user:{user_id}",
                user.to_dict(),
                ex=300  # 5 minutes
            )
            # Emit event
            await self.deps.producer.send(
                "user_accessed",
                {"user_id": user_id, "action": "read"}
            )
        
        return user
    
    async def create_user(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new user with event emission."""
        user = User(**user_data)
        self.deps.db.add(user)
        await self.deps.db.commit()
        
        # Emit event
        await self.deps.producer.send(
            "user_created",
            {"user_id": user.id, "data": user_data}
        )
        
        return user.to_dict()
```

---

## 3. Infrastructure

### Docker Configuration
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-dev

# Copy application
COPY src/ ./src/

# Set environment variables
ENV PYTHONPATH=/app

# Run service
CMD ["uvicorn", "src.service_name.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://user:pass@db:5432/dbname
      - REDIS_URL=redis://cache:6379/0
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on:
      - db
      - cache
      - kafka
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=dbname
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  cache:
    image: redis:6
    volumes:
      - redis_data:/data
  
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      - KAFKA_ADVERTISED_LISTENERS=PLAINTEXT://kafka:9092
      - KAFKA_ZOOKEEPER_CONNECT=zookeeper:2181
    depends_on:
      - zookeeper

volumes:
  postgres_data:
  redis_data:
```

---

## 4. Resilience

### Circuit Breaking
```python
from typing import Callable, Any
from functools import wraps
import time
import asyncio

class CircuitBreaker:
    """Circuit breaker implementation."""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: int = 60
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"
    
    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "half-open"
                else:
                    raise Exception("Circuit breaker is open")
            
            try:
                result = await func(*args, **kwargs)
                if self.state == "half-open":
                    self.state = "closed"
                    self.failures = 0
                return result
            except Exception as e:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.failure_threshold:
                    self.state = "open"
                
                raise e
        
        return wrapper

# Usage example
class ExternalService:
    @CircuitBreaker(failure_threshold=3, reset_timeout=30)
    async def make_request(self, url: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                return await response.json()
```

### Rate Limiting
```python
from typing import Dict, Optional
import time
import asyncio
from redis import Redis

class RateLimiter:
    """Rate limiter using Redis."""
    
    def __init__(
        self,
        redis: Redis,
        key_prefix: str,
        limit: int,
        window: int
    ):
        self.redis = redis
        self.key_prefix = key_prefix
        self.limit = limit
        self.window = window
    
    async def is_allowed(self, key: str) -> bool:
        """Check if request is allowed."""
        redis_key = f"{self.key_prefix}:{key}"
        current = int(time.time())
        window_start = current - self.window
        
        pipeline = self.redis.pipeline()
        pipeline.zremrangebyscore(redis_key, 0, window_start)
        pipeline.zadd(redis_key, {str(current): current})
        pipeline.zcard(redis_key)
        pipeline.expire(redis_key, self.window)
        _, _, count, _ = pipeline.execute()
        
        return count <= self.limit

class APIRateLimiter:
    """API rate limiting middleware."""
    
    def __init__(
        self,
        redis: Redis,
        limit: int = 100,
        window: int = 60
    ):
        self.limiter = RateLimiter(
            redis,
            "rate_limit",
            limit,
            window
        )
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        client_ip = request.client.host
        
        if not await self.limiter.is_allowed(client_ip):
            raise HTTPException(
                status_code=429,
                detail="Too many requests"
            )
        
        return await call_next(request)
```

---

## 5. Operations

### Monitoring & Logging
```python
import logging
import structlog
from prometheus_client import Counter, Histogram
import time
from typing import Callable, Any

# Metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

# Logging setup
logger = structlog.get_logger()

class MetricsMiddleware:
    """Middleware for collecting metrics."""
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        method = request.method
        path = request.url.path
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = response.status_code
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=status
            ).inc()
        except Exception as e:
            status = 500
            REQUEST_COUNT.labels(
                method=method,
                endpoint=path,
                status=status
            ).inc()
            raise e
        finally:
            duration = time.time() - start_time
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=path
            ).observe(duration)
        
        return response

class LoggingMiddleware:
    """Middleware for structured logging."""
    
    async def __call__(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        start_time = time.time()
        
        try:
            response = await call_next(request)
            status = response.status_code
            duration = time.time() - start_time
            
            logger.info(
                "request_processed",
                method=request.method,
                path=request.url.path,
                status=status,
                duration=duration,
                client_ip=request.client.host
            )
            
            return response
        except Exception as e:
            logger.error(
                "request_failed",
                method=request.method,
                path=request.url.path,
                error=str(e),
                client_ip=request.client.host
            )
            raise e
```

---

## Best Practices

1. **Service Design**
   - Keep services focused and small
   - Design clear API contracts
   - Implement proper error handling
   - Use async operations where appropriate
   - Follow domain-driven design

2. **Data Management**
   - Use appropriate databases
   - Implement caching strategies
   - Handle data consistency
   - Manage transactions properly
   - Monitor data performance

3. **Communication**
   - Use appropriate protocols
   - Implement circuit breakers
   - Handle timeouts properly
   - Use message queues
   - Document APIs clearly

4. **Deployment**
   - Use container orchestration
   - Implement CI/CD pipelines
   - Monitor service health
   - Scale automatically
   - Handle failures gracefully

5. **Security**
   - Implement authentication
   - Use secure communications
   - Monitor for threats
   - Handle sensitive data properly
   - Regular security updates

---

## Conclusion

Following these microservices standards ensures:
- Scalable architecture
- Reliable services
- Maintainable codebase
- Efficient operations
- Secure implementation

Remember to:
- Keep services independent
- Monitor performance
- Handle failures gracefully
- Document everything
- Maintain security

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at http://www.apache.org/licenses/LICENSE-2.0.
