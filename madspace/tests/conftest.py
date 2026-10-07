import os


def pytest_report_header(config):
    return f"madspace SIMD mode: {os.environ.get('MADSPACE_SIMD_MODE', 'scalar')}"
