import os
import unittest
import functools
import time

from typing import Iterator, List, Optional, Union

from distutils.util import strtobool

from .utils import is_ms_available


USER = "__DUMMY_TRANSFORMERS_USER__"
TOKEN = "hf_94wBhPGp6KrrTH3KDchhKpRxZwd6dmHWLL"

def parse_flag_from_env(key, default=False):
    try:
        value = os.environ[key]
    except KeyError:
        # KEY isn't set, default to `default`.
        _value = default
    else:
        # KEY is set, convert it to True or False.
        try:
            _value = strtobool(value)
        except ValueError:
            # More values are supported, but let's keep the message simple.
            raise ValueError(f"If set, {key} must be yes or no.")
    return _value


def parse_int_from_env(key, default=None):
    try:
        value = os.environ[key]
    except KeyError:
        _value = default
    else:
        try:
            _value = int(value)
        except ValueError:
            raise ValueError(f"If set, {key} must be a int.")
    return _value

_run_slow_tests = parse_flag_from_env("RUN_SLOW", default=False)
_run_staging = parse_flag_from_env("HUGGINGFACE_CO_STAGING", default=False)

def require_ms(test_case):
    """
    Decorator marking a test that requires MindSpore.
    These tests are skipped when MindSpore isn't installed.
    """
    return unittest.skipUnless(is_ms_available(), "test requires MindSpore")(test_case)

def slow(test_case):
    """
    Decorator marking a test as slow.
    Slow tests are skipped by default. Set the RUN_SLOW environment variable to a truthy value to run them.
    """
    return unittest.skipUnless(_run_slow_tests, "test is slow")(test_case)

def is_staging_test(test_case):
    """
    Decorator marking a test as a staging test.
    Those tests will run using the staging environment of huggingface.co instead of the real model hub.
    """
    if not _run_staging:
        return unittest.skip("test is staging test")(test_case)
    else:
        try:
            import pytest  # We don't need a hard dependency on pytest in the main library
        except ImportError:
            return test_case
        else:
            return pytest.mark.is_staging_test()(test_case)

def is_flaky(max_attempts: int = 5, wait_before_retry: Optional[float] = None):
    """
    To decorate flaky tests. They will be retried on failures.
    Args:
        max_attempts (`int`, *optional*, defaults to 5):
            The maximum number of attempts to retry the flaky test.
        wait_before_retry (`float`, *optional*):
            If provided, will wait that number of seconds before retrying the test.
    """

    def decorator(test_func_ref):
        @functools.wraps(test_func_ref)
        def wrapper(*args, **kwargs):
            retry_count = 1

            while retry_count < max_attempts:
                try:
                    return test_func_ref(*args, **kwargs)

                except Exception as err:
                    print(f"Test failed with {err} at try {retry_count}/{max_attempts}.", file=sys.stderr)
                    if wait_before_retry is not None:
                        time.sleep(wait_before_retry)
                    retry_count += 1

            return test_func_ref(*args, **kwargs)

        return wrapper

    return decorator
