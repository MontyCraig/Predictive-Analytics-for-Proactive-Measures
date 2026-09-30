# Django Coding Standards and Best Practices

A comprehensive guide for building scalable, maintainable Django applications following enterprise-level best practices.

## Table of Contents

1. **Project Structure**

    - Project Layout

    - App Organization

    - Settings Management

    - URL Configuration

    - Static & Media Files

2. **Models & Database**

    - Model Design Principles

    - Field Choices & Validation

    - Model Managers & QuerySets

    - Database Optimization

    - Migrations Management

3. **Views & Templates**

    - Class-Based vs Function-Based Views

    - Template Organization

    - Context Processors

    - Forms & Form Handling

    - Django REST Framework Integration

4. **Security**

    - Authentication & Authorization

    - CSRF Protection

    - XSS Prevention

    - SQL Injection Prevention

    - Security Middleware

5. **Performance**

    - Database Query Optimization

    - Caching Strategies

    - Static Files Handling

    - Asynchronous Views

    - Profiling & Monitoring

---

## 1. Project Structure

### Project Layout

```python
my_project/
├── manage.py
├── requirements/
│   ├── base.txt
│   ├── local.txt
│   └── production.txt
├── config/
│   ├── settings/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── local.py
│   │   └── production.py
│   ├── urls.py
│   └── wsgi.py
├── apps/
│   ├── core/
│   ├── users/
│   └── [feature_apps]/
├── static/
├── media/
├── templates/
└── docs/

```text
### App Organization

- Keep apps small and focused on a single responsibility

- Use meaningful app names that reflect their purpose

- Follow the "apps" directory pattern for better organization

- Include app-specific templates, static files, and tests within each app

### Settings Management

```python

# settings/base.py

from pathlib import Path
from decouple import config

BASE_DIR = Path(__file__).resolve().parent.parent.parent

SECRET_KEY = config('DJANGO_SECRET_KEY')

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    # ... other Django apps


    # Third-party apps

    'rest_framework',
    'corsheaders',

    # Local apps

    'apps.core.apps.CoreConfig',
    'apps.users.apps.UsersConfig',
]

```text
### URL Configuration

```python

# config/urls.py

from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/', include('apps.api.urls')),
    path('', include('apps.core.urls')),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

```text
---

## 2. Models & Database

### Model Design Principles

```python
from django.db import models
from django.conf import settings

class BaseModel(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        abstract = True

class Product(BaseModel):
    name = models.CharField(max_length=255)
    description = models.TextField()
    price = models.DecimalField(max_digits=10, decimal_places=2)
    is_active = models.BooleanField(default=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['name']),
            models.Index(fields=['is_active', 'created_at']),
        ]

    def __str__(self):
        return self.name

```text
### QuerySet Best Practices

```python

# managers.py

from django.db import models

class ProductQuerySet(models.QuerySet):
    def active(self):
        return self.filter(is_active=True)

    def with_related(self):
        return self.select_related('category').prefetch_related('tags')

class ProductManager(models.Manager):
    def get_queryset(self):
        return ProductQuerySet(self.model, using=self._db)

    def active_with_related(self):
        return self.get_queryset().active().with_related()

```text
---

## 3. Views & Templates

### Class-Based Views

```python
from django.views.generic import ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin

class ProductListView(LoginRequiredMixin, ListView):
    model = Product
    template_name = 'products/list.html'
    context_object_name = 'products'
    paginate_by = 20

    def get_queryset(self):
        return Product.objects.active_with_related()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Products'
        return context

```text
### Template Organization

```html
<!-- templates/base.html -->

<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}{% endblock %}</title>
    {% block extra_css %}{% endblock %}
</head>
<body>
    {% include 'includes/header.html' %}

    <main>
        {% block content %}{% endblock %}
    </main>

    {% include 'includes/footer.html' %}
    {% block extra_js %}{% endblock %}
</body>
</html>

```text
---

## 4. Security

### Authentication & Authorization

```python

# settings/base.py

AUTH_USER_MODEL = 'users.User'

# Secure password settings

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 12,
        }
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

# Session security

SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
SECURE_SSL_REDIRECT = True
SECURE_HSTS_SECONDS = 31536000  # 1 year

SECURE_HSTS_INCLUDE_SUBDOMAINS = True
SECURE_HSTS_PRELOAD = True

```text
---

## 5. Performance

### Database Optimization

- Use `select_related()` for foreign keys

- Use `prefetch_related()` for many-to-many relationships

- Create appropriate indexes

- Use `django.db.models.F()` expressions for updates

- Implement database connection pooling

### Caching Strategy

```python

# settings/base.py

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.redis.RedisCache',
        'LOCATION': config('REDIS_URL'),
    }
}

# views.py

from django.views.decorators.cache import cache_page
from django.utils.decorators import method_decorator

@method_decorator(cache_page(60 * 15), name='dispatch')
class ProductListView(ListView):
    # ... view implementation

```text
### Asynchronous Views

```python
from django.views.decorators.http import require_http_methods
from asgiref.sync import sync_to_async
from django.http import JsonResponse

@require_http_methods(['GET'])
async def async_product_view(request, pk):
    product = await sync_to_async(Product.objects.get)(pk=pk)
    data = {
        'id': product.id,
        'name': product.name,
        'price': str(product.price),
    }
    return JsonResponse(data)

```text
---

## Testing

### Test Organization

```python

# tests/test_models.py

from django.test import TestCase
from django.core.exceptions import ValidationError

class ProductModelTest(TestCase):
    def setUp(self):
        self.product = Product.objects.create(
            name='Test Product',
            price=99.99
        )

    def test_product_creation(self):
        self.assertEqual(self.product.name, 'Test Product')
        self.assertEqual(self.product.price, 99.99)

    def test_product_str_representation(self):
        self.assertEqual(str(self.product), 'Test Product')

```text
### Factory Pattern for Tests

```python

# tests/factories.py

import factory
from faker import Faker

fake = Faker()

class UserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = 'users.User'

    username = factory.LazyFunction(lambda: fake.user_name())
    email = factory.LazyFunction(lambda: fake.email())
    is_active = True

```text
---

## Deployment

### Production Checklist

1. Set `DEBUG = False`

2. Configure proper logging

3. Use secure SSL/TLS settings

4. Set up proper static file serving

5. Configure database connection pooling

6. Enable caching

7. Set up monitoring and error tracking

### Environment Variables

```python

# .env.example

DJANGO_SETTINGS_MODULE=config.settings.production
DJANGO_SECRET_KEY=your-secret-key
DATABASE_URL=postgres://user:password@localhost:5432/dbname
REDIS_URL=redis://localhost:6379/1

```text
---

## Documentation

### API Documentation

```python
from rest_framework import viewsets
from drf_spectacular.utils import extend_schema

@extend_schema(
    tags=['products'],
    description='API endpoints for product management'
)
class ProductViewSet(viewsets.ModelViewSet):
    """
    ViewSet for viewing and editing products.

    list:
    Return a list of all products.

    retrieve:
    Return a specific product by ID.
    """
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

```text
---

## Conclusion

Following these Django coding standards ensures:

- Consistent and maintainable code across the project

- Secure and performant applications

- Scalable architecture that can grow with your needs

- Easy onboarding for new team members

- Efficient development and deployment processes

Remember to regularly review and update these standards as Django evolves and new best practices emerge.

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at <http://www.apache.org/licenses/LICENSE-2.0.>
