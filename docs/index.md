# langbench

![PyPI - Version](https://img.shields.io/pypi/v/langbench)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![PyPI - Downloads](https://img.shields.io/pypi/dw/langbench)
[![CI](https://github.com/micvitc/langbench/actions/workflows/ci.yaml/badge.svg)](https://github.com/micvitc/langbench/actions/workflows/ci.yaml)

`langbench` is an easy to use benchmarking library for langchain based LLM pipelines.

## Installation

```bash

pip install langbench

```

## Metrics

`langbench` provides the following metrics:

- `toxicity`: Toxicity of the generated text.
- `bias`: Bias of the generated text, including:
    - political
    - gender
    - racial
    - educational
    - nationality
    - religious
    - others
- `latency`: Latency of the generated text.

Reports are generated in the form of an html file.

## Documentation

The official documentation is available at [langbench](https://micvitc.github.io/langbench/).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.