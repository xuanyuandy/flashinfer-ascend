"""Explicit release builder; importing flashinfer.aot does not invoke it."""

from .build import main

if __name__ == "__main__":
    main()
