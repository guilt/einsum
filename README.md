# Einsum

Simple, readable [Einstein Summation](https://en.wikipedia.org/wiki/Einstein_notation) in Python.

```bash
# Install
pip install python-einsum
```

## Simple Example

```python
from einsum import Einsum
import numpy as np

tensorAb = np.ones((2, 3))  # ab
tensorBd = np.ones((3, 4))  # bd
tensorDc = np.ones((4, 5))  # dc
einsumOp = Einsum("(ab,(bd,dc->bc)->ac)", tensorAb, tensorBd, tensorDc)
```

## Features

- Easy to Read and Understand
- Supports a variety of Notations

## Development

```bash
# Clone and install
git clone https://github.com/guilt/einsum
cd einsum
pip install -e ".[dev]"

# Run tests
python -m unittest discover tests/ -v
```

## License

MIT License. See [License](LICENSE.md) for details.

## Feedback

Made with ❤️ by [Vibe coding](https://en.wikipedia.org/wiki/Vibe_coding).

* Authors: [Grok 3.0](https://grok.com/), [Claude Sonnet 4](https://anthropic.com/) and [Karthik Kumar Viswanathan](https://github.com/guilt)
* Web   : http://karthikkumar.org
* Email : me@karthikkumar.org
