# Eris v0.1: A Novel LLM Evaluation Framework Using Debate Simulations

Eris is an innovative open-source framework for evaluating Large Language Models (LLMs) through simulated debates. Named after the Greek goddess of strife and discord, Eris pits different LLMs against each other in structured debates on complex topics to assess their reasoning, knowledge, and communication skills.

## Overview

Eris v0.1 is a proof-of-concept that:
- Simulates full debate flows between LLMs
- Evaluates debates using a separate LLM judge
- Analyzes results to produce comparative metrics

### **Eris v0.1 is a rough proof of concept and not meant to be used as a source of ground truth. It's an experimental benchmark designed to spark discussion and inspire new approaches to LLM evaluation.**

## Key Features

- Debate simulation: Constructive speeches, cross-examinations, rebuttals, and closing arguments
- Flexible evaluation: Customizable judging criteria
- Comprehensive analysis: Win rates and head-to-head performance metrics

## Installation

```bash
git clone https://github.com/AarushSah/eris.git
cd eris
pip install -r requirements.txt
```

## Configuration

Edit `main.py` to customize:
- List of LLMs to evaluate
- Debate topics
- Evaluation criteria

## Results

Initial results and visualizations can be found in the `results/` directory.

## Contributing

We welcome contributions! Please take a look at the Future Work section below.

## Limitations

Eris v0.1 is a proof-of-concept with known limitations:
- Potential judging bias
- One-size-fits-all prompting
- Limited evaluation criteria

We're actively working on addressing these in future versions.

## Future Work

Planned improvements for Eris v1.0 include:
- Ensemble judging
- Custom prompts per model
- Expanded evaluation criteria
- Larger sample sizes
- Human validation

## Acknowledgements

- OpenRouter for API access and grant support
- Weights & Biases for the Weave library

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Funding

If you're excited about Eris and would like to support this work, I'm currently seeking funding and grants to further develop and expand the project. If you're interested in supporting Eris or collaborating on building the 1.0 version, please don't hesitate to DM me (@AarushSah_) on Twitter.

## Contact

For questions or collaboration opportunities, please open an issue or contact @AarushSah_ on Twitter.