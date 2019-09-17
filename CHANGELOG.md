# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

Future releases will be following the pattern MAJOR.MINOR.PATCH, where:

MAJOR versions will differ in the selection of the benchmark sets  
MINOR versions signify changes that could affect the results  
PATCHES will not affect the benchmark results

All results obtained with the same MAJOR.MINOR version of DEEPOBS will be directly comparable.

## [Unreleased] - Version 1.2.0
### Added
- Changelog, documenting all current and future changes.
### Changed
### Removed
### Fixed

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


[Unreleased]: https://github.com/fsschneider/DeepOBS/compare/v1.1.1...version-1.2.0

[1.1.1]: https://github.com/fsschneider/DeepOBS/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/fsschneider/DeepOBS/releases/tag/v1.1.0
