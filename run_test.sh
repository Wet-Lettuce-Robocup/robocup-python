#!/bin/bash
cd "$(dirname "$0")"
PYTHONPATH=src .venv/bin/python tests/test_servo_positions.py