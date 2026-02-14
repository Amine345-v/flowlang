from setuptools import setup, find_packages

setup(
    name="flowlang",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "lark",
        "pydantic>=2.0",
        "networkx>=3.0",
        "loguru>=0.7",
        "openai",
        "anthropic",
        "google-generativeai",
        "mistralai",
        "cohere",
        "azure-identity",
        "opentelemetry-api",
        "opentelemetry-sdk",
    ],
    entry_points={
        "console_scripts": [
            "flow=flow:main",
        ],
    },
    python_requires=">=3.8",
)
