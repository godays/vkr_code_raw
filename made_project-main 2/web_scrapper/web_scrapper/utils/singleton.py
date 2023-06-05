#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Wrapper for implementation Singleton pattern
"""

from typing import TypeVar, Type, Any

C = TypeVar('C', bound=Type[Any])

def singleton(class_: C) -> C:
    """
    Wrapper for saving singleton instance
    """
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]

    return getinstance
