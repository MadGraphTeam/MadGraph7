# SPDX-License-Identifier: 0BSD
from __future__ import annotations

import io
import logging

from pyamplicol.reporting import (
    configure_cli_logging,
    get_logger,
    reset_cli_logging,
)


def test_cli_logging_is_package_local_idempotent_and_resettable() -> None:
    root = logging.getLogger()
    root_handlers = tuple(root.handlers)
    root_level = root.level
    stream = io.StringIO()
    reset_cli_logging()
    try:
        logger = configure_cli_logging("warning", stream=stream)
        configure_cli_logging("warning", stream=stream)
        owned = [
            handler
            for handler in logger.handlers
            if getattr(handler, "_pyamplicol_cli_handler", False)
        ]
        assert len(owned) == 1
        get_logger("unit").warning("diagnostic")
        assert "diagnostic" in stream.getvalue()
        assert tuple(root.handlers) == root_handlers
        assert root.level == root_level
    finally:
        reset_cli_logging()
    assert not any(
        getattr(handler, "_pyamplicol_cli_handler", False)
        for handler in get_logger().handlers
    )


def test_cli_logging_restores_embedding_application_state() -> None:
    logger = get_logger()
    reset_cli_logging()
    original_level = logger.level
    original_propagate = logger.propagate
    external_handler = logging.NullHandler()
    logger.addHandler(external_handler)
    logger.setLevel(logging.ERROR)
    logger.propagate = False
    try:
        configure_cli_logging(logging.INFO, stream=io.StringIO())
        assert logger.level == logging.INFO
        assert logger.propagate is False

        reset_cli_logging()

        assert logger.level == logging.ERROR
        assert logger.propagate is False
        assert external_handler in logger.handlers
    finally:
        reset_cli_logging()
        logger.removeHandler(external_handler)
        logger.setLevel(original_level)
        logger.propagate = original_propagate
