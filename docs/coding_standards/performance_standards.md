# Python Performance Standards and Best Practices

A comprehensive guide for optimizing Python applications, focusing on performance bottlenecks, profiling, and optimization techniques.

## Table of Contents

1. **Performance Profiling**
    - Code Profiling
    - Memory Profiling
    - CPU Profiling
    - I/O Profiling
    - Bottleneck Analysis

2. **Code Optimization**
    - Data Structures
    - Algorithms
    - Memory Management
    - CPU Utilization
    - I/O Operations

3. **Concurrency & Parallelism**
    - Threading
    - Multiprocessing
    - Async/Await
    - Worker Pools
    - Distributed Computing

4. **Memory Management**
    - Memory Usage
    - Garbage Collection
    - Memory Leaks
    - Caching
    - Resource Pooling

5. **Performance Testing**
    - Load Testing
    - Stress Testing
    - Benchmarking
    - Monitoring
    - Reporting

---

## 1. Performance Profiling

### Code Profiling Tools
python
import cProfile
import pstats
from line_profiler import LineProfiler
from memory_profiler import profile as memory_profile
import time
from functools import wraps

def timing_decorator(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__} took {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def profile_function(func):
    """Profile function execution using cProfile."""
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        try:
            return profiler.runcall(func, *args, **kwargs)
        finally:
            stats = pstats.Stats(profiler)
            stats.sort_stats('cumulative')
            stats.print_stats()
    return wrapper

# Usage example
@timing_decorator
@profile_function
def expensive_operation():
    return sum(i * i for i in range(1000000))
```

### Memory Profiling
```python
from memory_profiler import profile
import psutil
import os

class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def memory_check(self, threshold_mb=100):
        """Check if memory usage exceeds threshold."""
        usage = self.get_memory_usage()
        if usage > threshold_mb:
            print(f"Warning: Memory usage ({usage:.2f}MB) exceeds threshold")
        return usage

@profile
def memory_intensive_function(size: int):
    """Example of memory-profiled function."""
    # Create large list
    data = [i * i for i in range(size)]
    # Process data
    result = sum(data)
    return result

# Memory-efficient version
def memory_efficient_function(size: int):
    """Memory-efficient version using generator."""
    return sum(i * i for i in range(size))
```

---

## 2. Code Optimization

### Data Structure Optimization
```python
from collections import defaultdict, deque
from typing import List, Dict, Set, Any
import array

class OptimizedCollections:
    @staticmethod
    def efficient_list_operations():
        """Demonstrate efficient list operations."""
        # Use list comprehension instead of map/filter
        squares = [x * x for x in range(1000)]
        
        # Use array for numeric data
        numbers = array.array('i', (x for x in range(1000)))
        
        # Use deque for queue operations
        queue = deque(maxlen=1000)
        
        # Use set for membership testing
        lookup_set = set(squares)
        
        return squares, numbers, queue, lookup_set
    
    @staticmethod
    def efficient_dict_operations():
        """Demonstrate efficient dictionary operations."""
        # Use defaultdict for automatic default values
        counts = defaultdict(int)
        
        # Use dict comprehension
        squares_dict = {x: x * x for x in range(100)}
        
        # Use dict.setdefault for initialization
        groups: Dict[str, List[str]] = {}
        for key, value in [("A", "1"), ("B", "2"), ("A", "3")]:
            groups.setdefault(key, []).append(value)
        
        return counts, squares_dict, groups

class StringOptimization:
    @staticmethod
    def efficient_string_operations(strings: List[str]) -> str:
        """Demonstrate efficient string operations."""
        # Use join instead of += for string concatenation
        result = " ".join(strings)
        
        # Use string methods instead of regex for simple cases
        cleaned = result.strip().lower()
        
        # Use string formatting efficiently
        formatted = f"{cleaned[:10]}..."
        
        return formatted
```

### Algorithm Optimization
```python
from typing import List, Optional
import heapq
from functools import lru_cache

class AlgorithmOptimization:
    @staticmethod
    @lru_cache(maxsize=128)
    def fibonacci(n: int) -> int:
        """Optimized Fibonacci using memoization."""
        if n < 2:
            return n
        return AlgorithmOptimization.fibonacci(n - 1) + AlgorithmOptimization.fibonacci(n - 2)
    
    @staticmethod
    def efficient_search(sorted_data: List[int], target: int) -> Optional[int]:
        """Binary search instead of linear search."""
        left, right = 0, len(sorted_data) - 1
        
        while left <= right:
            mid = (left + right) // 2
            if sorted_data[mid] == target:
                return mid
            elif sorted_data[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        
        return None
    
    @staticmethod
    def top_k_elements(data: List[int], k: int) -> List[int]:
        """Efficient top-k using heap."""
        return heapq.nlargest(k, data)
```

---

## 3. Concurrency & Parallelism

### Async Operations
```python
import asyncio
from typing import List
import aiohttp
import time

class AsyncOperations:
    def __init__(self):
        self.session = None
    
    async def setup(self):
        """Setup async session."""
        self.session = aiohttp.ClientSession()
    
    async def cleanup(self):
        """Cleanup async resources."""
        if self.session:
            await self.session.close()
    
    async def fetch_url(self, url: str) -> dict:
        """Fetch single URL asynchronously."""
        async with self.session.get(url) as response:
            return await response.json()
    
    async def fetch_all_urls(self, urls: List[str]) -> List[dict]:
        """Fetch multiple URLs concurrently."""
        tasks = [self.fetch_url(url) for url in urls]
        return await asyncio.gather(*tasks)

class ParallelProcessing:
    @staticmethod
    def cpu_intensive_task(data: List[int]) -> List[int]:
        """CPU-intensive task for parallel processing."""
        return [x * x for x in data]
    
    @staticmethod
    def parallel_processing(data: List[int], chunks: int = 4) -> List[int]:
        """Process data in parallel using multiprocessing."""
        from multiprocessing import Pool
        
        chunk_size = len(data) // chunks
        with Pool(processes=chunks) as pool:
            results = pool.map(
                ParallelProcessing.cpu_intensive_task,
                [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
            )
        
        return [item for sublist in results for item in sublist]
```

---

## 4. Memory Management

### Resource Management
```python
from typing import Any, Optional
from contextlib import contextmanager
import gc

class ResourceManager:
    def __init__(self):
        self.resources: Dict[str, Any] = {}
    
    @contextmanager
    def resource_context(self, name: str, resource: Any):
        """Context manager for resource handling."""
        try:
            self.resources[name] = resource
            yield resource
        finally:
            self.cleanup_resource(name)
    
    def cleanup_resource(self, name: str):
        """Clean up specific resource."""
        if name in self.resources:
            del self.resources[name]
            gc.collect()  # Force garbage collection

class CacheManager:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache."""
        return self.cache.get(key)
    
    def set(self, key: str, value: Any):
        """Set item in cache with size management."""
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value
```

---

## 5. Performance Testing

### Load Testing
```python
import time
import statistics
from typing import List, Callable, Any
import concurrent.futures

class PerformanceTester:
    def __init__(self):
        self.results: List[float] = []
    
    def measure_execution(
        self,
        func: Callable,
        *args,
        iterations: int = 1000
    ) -> dict:
        """Measure function execution times."""
        self.results = []
        
        for _ in range(iterations):
            start_time = time.perf_counter()
            func(*args)
            end_time = time.perf_counter()
            self.results.append(end_time - start_time)
        
        return {
            'min': min(self.results),
            'max': max(self.results),
            'avg': statistics.mean(self.results),
            'median': statistics.median(self.results),
            'std_dev': statistics.stdev(self.results)
        }
    
    def load_test(
        self,
        func: Callable,
        *args,
        concurrent_users: int = 10,
        requests_per_user: int = 100
    ) -> dict:
        """Simulate concurrent users."""
        start_time = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=concurrent_users
        ) as executor:
            futures = [
                executor.submit(func, *args)
                for _ in range(concurrent_users * requests_per_user)
            ]
            concurrent.futures.wait(futures)
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        return {
            'total_time': total_time,
            'requests_per_second': (
                concurrent_users * requests_per_user
            ) / total_time,
            'average_response_time': total_time / (
                concurrent_users * requests_per_user
            )
        }
```

---

## Best Practices

1. **Code Optimization**
   - Use appropriate data structures
   - Implement efficient algorithms
   - Minimize memory usage
   - Optimize I/O operations
   - Use built-in functions

2. **Concurrency**
   - Choose appropriate concurrency model
   - Handle shared resources properly
   - Implement proper error handling
   - Monitor resource usage
   - Test concurrent operations

3. **Memory Management**
   - Monitor memory usage
   - Implement proper cleanup
   - Use context managers
   - Handle large datasets efficiently
   - Implement caching strategically

4. **Performance Testing**
   - Implement comprehensive tests
   - Monitor system resources
   - Test under various loads
   - Measure key metrics
   - Document performance requirements

5. **Monitoring**
   - Track key performance indicators
   - Monitor resource usage
   - Set up alerting
   - Collect performance metrics
   - Analyze trends

---

## Conclusion

Following these performance standards ensures:
- Efficient code execution
- Optimal resource usage
- Scalable applications
- Reliable performance
- Maintainable optimizations

Remember to:
- Profile before optimizing
- Test performance regularly
- Monitor resource usage
- Document optimizations
- Maintain balance between performance and readability

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at http://www.apache.org/licenses/LICENSE-2.0.
