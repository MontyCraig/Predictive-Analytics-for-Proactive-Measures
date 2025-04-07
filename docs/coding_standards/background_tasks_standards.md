# Python Background Tasks Standards and Best Practices

A comprehensive guide for implementing background tasks, job queues, and asynchronous processing in Python applications.

## Table of Contents

1. **Task Processing**
    - Celery Tasks
    - RQ (Redis Queue)
    - Background Workers
    - Task Scheduling
    - Task Monitoring

2. **Task Design**
    - Task Structure
    - Error Handling
    - Retries & Backoff
    - Task Priority
    - Task Dependencies

3. **Task Queue**
    - Queue Management
    - Message Brokers
    - Result Backends
    - Queue Monitoring
    - Dead Letter Queues

4. **Task Scheduling**
    - Periodic Tasks
    - Cron Jobs
    - Dynamic Scheduling
    - Task Distribution
    - Schedule Management

5. **Task Monitoring**
    - Task Status
    - Performance Metrics
    - Error Tracking
    - Resource Usage
    - Alerting

---

## 1. Task Processing

### Celery Configuration
```python
from celery import Celery
from typing import Any, Dict, Optional
from datetime import timedelta
import logging

# Celery configuration
app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/1'
)

app.conf.update(
    # Task settings
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # Task execution settings
    task_time_limit=3600,  # 1 hour
    task_soft_time_limit=3300,  # 55 minutes
    
    # Worker settings
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # Result backend settings
    result_expires=timedelta(days=1),
    
    # Retry settings
    task_default_retry_delay=300,  # 5 minutes
    task_max_retries=3,
    
    # Logging
    worker_log_format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s',
    worker_task_log_format='[%(asctime)s: %(levelname)s/%(processName)s] [%(task_name)s(%(task_id)s)] %(message)s'
)

# Task base class
class BaseTask(app.Task):
    """Base task with error handling and logging."""
    
    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any
    ):
        """Handle task failure."""
        logging.error(
            f"Task {task_id} failed: {exc}",
            exc_info=einfo
        )
    
    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any
    ):
        """Handle task retry."""
        logging.warning(
            f"Task {task_id} retrying: {exc}",
            exc_info=einfo
        )
    
    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple,
        kwargs: dict
    ):
        """Handle task success."""
        logging.info(f"Task {task_id} completed successfully")

# Task implementation
@app.task(
    base=BaseTask,
    bind=True,
    max_retries=3,
    default_retry_delay=300
)
def process_data(
    self,
    data: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Example task with retry logic."""
    try:
        # Process data
        result = perform_processing(data)
        return result
    except Exception as exc:
        # Handle specific exceptions differently
        if isinstance(exc, ValueError):
            # Don't retry for validation errors
            raise
        
        # Retry with exponential backoff
        retry_delay = self.default_retry_delay * (2 ** self.request.retries)
        raise self.retry(exc=exc, countdown=retry_delay)
```

### Task Chains and Groups
```python
from celery import chain, group, chord
from typing import List, Any

# Task chain
@app.task
def fetch_data(url: str) -> Dict[str, Any]:
    """Fetch data from URL."""
    return {"url": url, "data": "fetched_data"}

@app.task
def process_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Process fetched data."""
    return {"processed": data}

@app.task
def save_results(data: Dict[str, Any]) -> None:
    """Save processed data."""
    pass

def process_url(url: str):
    """Chain multiple tasks."""
    return chain(
        fetch_data.s(url),
        process_data.s(),
        save_results.s()
    )()

# Task groups
@app.task
def process_batch(urls: List[str]) -> List[Dict[str, Any]]:
    """Process multiple URLs in parallel."""
    job = group(
        fetch_data.s(url) for url in urls
    )
    return job.apply_async()

# Task chord
@app.task
def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize multiple task results."""
    return {
        "total": len(results),
        "successful": sum(1 for r in results if r.get("success"))
    }

def process_urls_with_summary(urls: List[str]):
    """Process URLs and summarize results."""
    return chord(
        (fetch_data.s(url) for url in urls),
        summarize_results.s()
    )()
```

---

## 2. Task Design

### Task Structure
```python
from typing import Optional, Dict, Any
from datetime import datetime
import json
import logging
from contextlib import contextmanager

class TaskContext:
    """Context for task execution."""
    
    def __init__(
        self,
        task_id: str,
        retry_count: int = 0,
        parent_id: Optional[str] = None
    ):
        self.task_id = task_id
        self.retry_count = retry_count
        self.parent_id = parent_id
        self.start_time = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            "task_id": self.task_id,
            "retry_count": self.retry_count,
            "parent_id": self.parent_id,
            "start_time": self.start_time.isoformat()
        }

@contextmanager
def task_context(task_id: str):
    """Context manager for task execution."""
    context = TaskContext(task_id)
    try:
        yield context
    finally:
        duration = (datetime.utcnow() - context.start_time).total_seconds()
        logging.info(
            f"Task {task_id} completed in {duration:.2f}s"
        )

class TaskResult:
    """Task execution result."""
    
    def __init__(
        self,
        success: bool,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        self.success = success
        self.data = data
        self.error = error
        self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp.isoformat()
        }
    
    @classmethod
    def success(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create success result."""
        return cls(True, data=data)
    
    @classmethod
    def error(cls, error: str) -> 'TaskResult':
        """Create error result."""
        return cls(False, error=error)

@app.task(bind=True)
def example_task(
    self,
    data: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Example task with proper structure."""
    with task_context(self.request.id) as context:
        try:
            # Process data
            result = process_data(data)
            return TaskResult.success(result).to_dict()
        except Exception as e:
            error_msg = f"Task failed: {str(e)}"
            logging.error(error_msg, exc_info=True)
            return TaskResult.error(error_msg).to_dict()
```

---

## 3. Task Queue

### Queue Management
```python
from typing import Optional, List, Dict, Any
from redis import Redis
import json
from datetime import datetime, timedelta

class QueueManager:
    """Manage task queues."""
    
    def __init__(
        self,
        redis: Redis,
        default_queue: str = "default"
    ):
        self.redis = redis
        self.default_queue = default_queue
    
    def enqueue(
        self,
        task_name: str,
        args: tuple,
        kwargs: dict,
        queue: Optional[str] = None,
        priority: int = 0
    ) -> str:
        """Enqueue task with priority."""
        queue_name = queue or self.default_queue
        task_id = generate_task_id()
        
        task_data = {
            "id": task_id,
            "name": task_name,
            "args": args,
            "kwargs": kwargs,
            "priority": priority,
            "enqueued_at": datetime.utcnow().isoformat()
        }
        
        # Use sorted set for priority queue
        score = priority * -1  # Higher priority = lower score
        self.redis.zadd(
            f"queue:{queue_name}",
            {json.dumps(task_data): score}
        )
        
        return task_id
    
    def dequeue(
        self,
        queue: Optional[str] = None,
        timeout: int = 30
    ) -> Optional[Dict[str, Any]]:
        """Dequeue task with highest priority."""
        queue_name = queue or self.default_queue
        
        # Get highest priority task
        result = self.redis.zpopmin(f"queue:{queue_name}")
        if not result:
            return None
        
        task_data = json.loads(result[0][0])
        return task_data
    
    def get_queue_length(
        self,
        queue: Optional[str] = None
    ) -> int:
        """Get queue length."""
        queue_name = queue or self.default_queue
        return self.redis.zcard(f"queue:{queue_name}")
    
    def clear_queue(
        self,
        queue: Optional[str] = None
    ) -> int:
        """Clear queue."""
        queue_name = queue or self.default_queue
        return self.redis.delete(f"queue:{queue_name}")

class DeadLetterQueue:
    """Manage failed tasks."""
    
    def __init__(self, redis: Redis):
        self.redis = redis
    
    def add_failed_task(
        self,
        task_data: Dict[str, Any],
        error: str
    ) -> None:
        """Add failed task to DLQ."""
        failed_task = {
            **task_data,
            "error": error,
            "failed_at": datetime.utcnow().isoformat()
        }
        
        self.redis.lpush(
            "dead_letter_queue",
            json.dumps(failed_task)
        )
    
    def get_failed_tasks(
        self,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get failed tasks."""
        tasks = self.redis.lrange(
            "dead_letter_queue",
            0,
            limit - 1
        )
        return [json.loads(t) for t in tasks]
    
    def retry_failed_task(
        self,
        task_data: Dict[str, Any],
        queue_manager: QueueManager
    ) -> str:
        """Retry failed task."""
        return queue_manager.enqueue(
            task_data["name"],
            task_data["args"],
            task_data["kwargs"],
            priority=1  # Higher priority for retries
        )
```

---

## 4. Task Scheduling

### Periodic Tasks
```python
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import croniter
import json
from redis import Redis

class TaskScheduler:
    """Schedule periodic tasks."""
    
    def __init__(self, redis: Redis):
        self.redis = redis
    
    def schedule_task(
        self,
        task_name: str,
        schedule: str,  # Cron expression
        args: tuple = (),
        kwargs: dict = None,
        options: Optional[Dict[str, Any]] = None
    ) -> str:
        """Schedule periodic task."""
        schedule_id = generate_schedule_id()
        
        schedule_data = {
            "id": schedule_id,
            "task_name": task_name,
            "schedule": schedule,
            "args": args,
            "kwargs": kwargs or {},
            "options": options or {},
            "last_run": None,
            "next_run": None,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Calculate next run
        cron = croniter.croniter(schedule, datetime.utcnow())
        schedule_data["next_run"] = cron.get_next(datetime).isoformat()
        
        # Store schedule
        self.redis.hset(
            "task_schedules",
            schedule_id,
            json.dumps(schedule_data)
        )
        
        return schedule_id
    
    def get_due_tasks(self) -> List[Dict[str, Any]]:
        """Get tasks due for execution."""
        now = datetime.utcnow()
        due_tasks = []
        
        # Get all schedules
        schedules = self.redis.hgetall("task_schedules")
        for schedule_id, schedule_json in schedules.items():
            schedule = json.loads(schedule_json)
            next_run = datetime.fromisoformat(schedule["next_run"])
            
            if next_run <= now:
                due_tasks.append(schedule)
                
                # Calculate next run
                cron = croniter.croniter(
                    schedule["schedule"],
                    now
                )
                schedule["last_run"] = now.isoformat()
                schedule["next_run"] = cron.get_next(
                    datetime
                ).isoformat()
                
                # Update schedule
                self.redis.hset(
                    "task_schedules",
                    schedule_id,
                    json.dumps(schedule)
                )
        
        return due_tasks
    
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove task schedule."""
        return bool(
            self.redis.hdel("task_schedules", schedule_id)
        )

class SchedulerWorker:
    """Worker for executing scheduled tasks."""
    
    def __init__(
        self,
        scheduler: TaskScheduler,
        queue_manager: QueueManager
    ):
        self.scheduler = scheduler
        self.queue_manager = queue_manager
    
    async def run(self):
        """Run scheduler worker."""
        while True:
            try:
                # Get due tasks
                due_tasks = self.scheduler.get_due_tasks()
                
                # Enqueue tasks
                for task in due_tasks:
                    self.queue_manager.enqueue(
                        task["task_name"],
                        task["args"],
                        task["kwargs"],
                        options=task["options"]
                    )
                
                # Wait before next check
                await asyncio.sleep(60)
            except Exception as e:
                logging.error(
                    f"Scheduler error: {str(e)}",
                    exc_info=True
                )
                await asyncio.sleep(5)
```

---

## 5. Task Monitoring

### Monitoring System
```python
from typing import Dict, Any, List
from datetime import datetime, timedelta
import psutil
import json
from redis import Redis

class TaskMonitor:
    """Monitor task execution and performance."""
    
    def __init__(self, redis: Redis):
        self.redis = redis
    
    def record_task_start(
        self,
        task_id: str,
        task_name: str
    ) -> None:
        """Record task start."""
        task_data = {
            "id": task_id,
            "name": task_name,
            "start_time": datetime.utcnow().isoformat(),
            "status": "running"
        }
        
        self.redis.hset(
            "running_tasks",
            task_id,
            json.dumps(task_data)
        )
    
    def record_task_completion(
        self,
        task_id: str,
        success: bool,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Record task completion."""
        # Get task data
        task_json = self.redis.hget("running_tasks", task_id)
        if not task_json:
            return
        
        task_data = json.loads(task_json)
        task_data.update({
            "end_time": datetime.utcnow().isoformat(),
            "status": "completed" if success else "failed",
            "result": result,
            "error": error
        })
        
        # Move to completed tasks
        self.redis.hdel("running_tasks", task_id)
        self.redis.hset(
            "completed_tasks",
            task_id,
            json.dumps(task_data)
        )
        
        # Update metrics
        if success:
            self.redis.hincrby("task_metrics", "success_count", 1)
        else:
            self.redis.hincrby("task_metrics", "failure_count", 1)
    
    def get_task_metrics(
        self,
        window: timedelta = timedelta(hours=1)
    ) -> Dict[str, Any]:
        """Get task execution metrics."""
        now = datetime.utcnow()
        cutoff = now - window
        
        # Get completed tasks in window
        completed_tasks = self.redis.hgetall("completed_tasks")
        tasks_in_window = [
            json.loads(t) for t in completed_tasks.values()
            if datetime.fromisoformat(
                json.loads(t)["end_time"]
            ) > cutoff
        ]
        
        # Calculate metrics
        total_tasks = len(tasks_in_window)
        successful_tasks = sum(
            1 for t in tasks_in_window
            if t["status"] == "completed"
        )
        failed_tasks = total_tasks - successful_tasks
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": (
                successful_tasks / total_tasks
                if total_tasks > 0 else 0
            )
        }
    
    def get_resource_usage(self) -> Dict[str, float]:
        """Get system resource usage."""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
```

---

## Best Practices

1. **Task Design**
   - Keep tasks small and focused
   - Implement proper error handling
   - Use appropriate retry strategies
   - Handle task dependencies
   - Document task behavior

2. **Queue Management**
   - Monitor queue lengths
   - Implement dead letter queues
   - Handle priority properly
   - Manage resource usage
   - Clean up completed tasks

3. **Task Scheduling**
   - Use appropriate scheduling patterns
   - Handle timezone differences
   - Implement proper error recovery
   - Monitor schedule execution
   - Document scheduling logic

4. **Performance**
   - Monitor task execution time
   - Handle resource constraints
   - Implement proper caching
   - Optimize task execution
   - Scale workers appropriately

5. **Monitoring**
   - Track task status
   - Monitor resource usage
   - Implement proper logging
   - Set up alerting
   - Maintain task history

---

## Conclusion

Following these background tasks standards ensures:
- Reliable task execution
- Efficient resource usage
- Proper error handling
- Scalable task processing
- Maintainable task code

Remember to:
- Monitor task execution
- Handle errors gracefully
- Document task behavior
- Maintain task queues
- Scale appropriately

## License

This document is licensed under the Apache License, Version 2.0. You may obtain a copy of the license at http://www.apache.org/licenses/LICENSE-2.0.
