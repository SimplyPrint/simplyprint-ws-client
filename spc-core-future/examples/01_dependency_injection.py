"""
Example 1: Dependency Injection

Demonstrates the clean DI system with Injected marker.

Run: python -m examples.01_dependency_injection
"""

import asyncio
import logging
from dataclasses import dataclass
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from spc_core import Container, Injected, Service

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================================
# Example 1: Basic Dependency Injection
# ============================================================================

@dataclass
class DatabaseConfig:
    """Configuration for database"""
    host: str = "localhost"
    port: int = 5432
    database: str = "myapp"


class Database:
    """Simulated database connection"""

    def __init__(self, config: DatabaseConfig = Injected):
        self.config = config
        self.connected = False
        logger.info(f"Database created: {config.host}:{config.port}/{config.database}")

    async def connect(self):
        """Connect to database"""
        await asyncio.sleep(0.1)  # Simulate connection
        self.connected = True
        logger.info("Database connected")

    async def query(self, sql: str):
        """Execute query"""
        if not self.connected:
            raise RuntimeError("Not connected")
        logger.info(f"Executing: {sql}")
        return [{"id": 1, "name": "Test"}]


class UserRepository:
    """Repository for user data"""

    def __init__(self, db: Database = Injected):
        self.db = db
        logger.info("UserRepository created")

    async def get_users(self):
        """Get all users"""
        return await self.db.query("SELECT * FROM users")


class UserService:
    """Service layer for user operations"""

    def __init__(self, repo: UserRepository = Injected):
        self.repo = repo
        logger.info("UserService created")

    async def list_users(self):
        """List all users"""
        logger.info("Listing users...")
        users = await self.repo.get_users()
        for user in users:
            logger.info(f"  User: {user}")
        return users


# ============================================================================
# Example 2: Service with Lifecycle
# ============================================================================

class CacheService(Service):
    """Cache service with lifecycle hooks"""

    def __init__(self):
        self._cache = {}
        logger.info("CacheService created")

    async def start(self):
        """Called when service starts"""
        logger.info("CacheService starting...")
        await asyncio.sleep(0.1)
        logger.info("CacheService started")

    async def stop(self):
        """Called when service stops"""
        logger.info("CacheService stopping...")
        self._cache.clear()
        logger.info("CacheService stopped")

    def get(self, key: str):
        return self._cache.get(key)

    def set(self, key: str, value):
        logger.info(f"Cache set: {key} = {value}")
        self._cache[key] = value


class CachedUserService:
    """User service with caching"""

    def __init__(
        self,
        repo: UserRepository = Injected,
        cache: CacheService = Injected
    ):
        self.repo = repo
        self.cache = cache
        logger.info("CachedUserService created")

    async def get_user(self, user_id: int):
        """Get user with caching"""
        cache_key = f"user:{user_id}"

        # Check cache
        cached = self.cache.get(cache_key)
        if cached:
            logger.info(f"Cache hit: {cache_key}")
            return cached

        # Query database
        logger.info(f"Cache miss: {cache_key}")
        user = await self.repo.get_users()

        # Store in cache
        self.cache.set(cache_key, user)

        return user


# ============================================================================
# Example 3: Factory Functions
# ============================================================================

def create_logger():
    """Factory for creating loggers"""
    return logging.getLogger("app")


class LoggingService:
    """Service with custom factory"""

    def __init__(self, logger=Injected.of(create_logger)):
        self.logger = logger
        self.logger.info("LoggingService created with custom logger")

    def log(self, message: str):
        self.logger.info(f"[APP] {message}")


# ============================================================================
# Main Demo
# ============================================================================

async def main():
    print("=" * 70)
    print("  Example 1: Dependency Injection")
    print("=" * 70)
    print()

    # Create container
    container = Container()

    # Register config (singleton instance)
    config = DatabaseConfig(host="db.example.com", database="production")
    container.register_instance(DatabaseConfig, config)

    # Register services (container will auto-resolve dependencies)
    container.register(Database)
    container.register(UserRepository)
    container.register(UserService)
    container.register(CacheService)
    container.register(CachedUserService)
    container.register(LoggingService)

    print("--- Basic Dependency Injection ---\n")

    # Resolve UserService (automatically resolves Database and UserRepository)
    user_service = await container.resolve(UserService)

    # Connect database
    db = await container.resolve(Database)
    await db.connect()

    # Use service
    await user_service.list_users()

    print("\n--- Service Lifecycle ---\n")

    # Resolve service with lifecycle (start() automatically called)
    cached_service = await container.resolve(CachedUserService)

    # Use cached service
    await cached_service.get_user(1)
    await cached_service.get_user(1)  # Cache hit

    print("\n--- Custom Factory ---\n")

    # Service with custom factory
    logging_service = await container.resolve(LoggingService)
    logging_service.log("This uses a custom logger factory")

    print("\n--- Cleanup ---\n")

    # Shutdown (calls stop() on all Service instances)
    await container.shutdown()

    print("\n" + "=" * 70)
    print("  Demo Complete")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
