"""
Boilerplate async utils
"""

__all__ = ["cond_notify", "cond_notify_all", "cond_wait"]

import asyncio


async def cond_notify_all(cond: asyncio.Condition):
    async with cond:
        cond.notify_all()


async def cond_notify(cond: asyncio.Condition):
    async with cond:
        cond.notify()


async def cond_wait(cond: asyncio.Condition):
    async with cond:
        await cond.wait()
