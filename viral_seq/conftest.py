def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks the tests as slow (deselect with '-m \"not slow\"')"
    )
