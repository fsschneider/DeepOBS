# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Future releases will be following the pattern MAJOR.MINOR.PATCH, where:

MAJOR versions will differ in the selection of the benchmark sets  
MINOR versions signify changes that could affect the results  
PATCHES will not affect the benchmark results

All results obtained with the same MAJOR.MINOR version of DEEPOBS will be directly comparable.

## [Beta] - Version 1.2.0-beta0
### Added
- Changelog, documenting all current and future changes.
- Version info available via `deepobs.__version__`
- PyTorch support (currently not all test problems).
- A Tuning module automating the hyperparamter tuning process.
- Added a separate hold-out validation set for hyperparameter tuning. This reduces the size of the training data.
### Changed
- Refactored Analyzer module (more flexible and interpretable).
- Smaller training data set due to additional validation set.
- Runners break from the training loop if the loss becomes NaN.
- Runners now return the output dictionary.
- Additional training parameters can be passed as kwargs to the `run()` method.
- The small and large benchmark sets are now global variables.
- Default test problem settings (`batch size`, `num_epochs`,...) for are now global variables.
- `JSON` output is now dumped in human readable format.
- The accuracy is now only printed if available.
- Simplified the API of the Runner module.
- The runner with a learning rate schedule is now an extra class.
- Extra folder with extensive examples.
- Switched from `matplotlib2tikz` (discontinued) to `tikzplotlib`.
### Removed
- Currently no baselines! Changes to the training data set (smaller, due to the validation set) and other changed require us to recompute the baselines. We are currently doing a more extensive generation of the baseline, with many more optimizers.
- Currently no LaTeX Output
### Fixed
- `fmnist_mlp` was not using Fashion-MNIST but MNIST data.
- Training set of `SVHN` was limited to `64,000` examples. Now it is the full training set.
- Various other small bugfixes.
- `Numpy` is now also seeded.

## [1.1.1] - 2019-03-13
### Added
- MIT License
### Changed
- Setup.py to add more descriptions for PyPI.
- Do not ship the baselines with DeepOBS, but keep them in a [separate repo](https://github.com/fsschneider/DeepOBS_Baselines).
### Removed
- Removed baselines from package. They are now shipped separately.
### Fixed
- Fixed check for non-existing labels in test_quadratic.py [see this commit](https://github.com/fsschneider/DeepOBS/commit/2c287a89a9197a9880cbb00ff13516c128cc26f2).
- Fixed 

## [1.1.0] - 2019-03-01
### Added
- First release version of DeepOBS.


[Beta]: https://github.com/fsschneider/DeepOBS/compare/v1.1.1...v1.2.0-beta0

[1.1.1]: https://github.com/fsschneider/DeepOBS/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/fsschneider/DeepOBS/releases/tag/v1.1.0
