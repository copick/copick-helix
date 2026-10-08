# Changelog

## 0.1.0 (2026-10-08)


### Features

* actin family and the term route from tilt-series segments ([12a54df](https://github.com/copick/copick-helix/commit/12a54df60870909e5e076af216220b689da41086))
* **bands:** confident-filament excess over decoys as a second detection test ([dd75572](https://github.com/copick/copick-helix/commit/dd7557206479675d6d55dbcbce2a27878806c0e6))
* copick outputs and the helix-picks command ([39caca2](https://github.com/copick/copick-helix/commit/39caca2d8ff922332ec9d432e5388d067be19a81))
* copick-helix core and the microtubule family ([6f64702](https://github.com/copick/copick-helix/commit/6f647022efdfa488ca5b226b7b56e03014b3bb74))
* helix-recentre, and intermediate filaments recentred only ([4f81294](https://github.com/copick/copick-helix/commit/4f812944a8b893a6d36b7647b59b9137f294d131))
* intermediate filament family, generic helical families and a lattice gate ([bbae16f](https://github.com/copick/copick-helix/commit/bbae16fbd3018af6a2e896207d72dd2727f51db1))
* per-segment registration and lattice-registered particles ([9a0e0ce](https://github.com/copick/copick-helix/commit/9a0e0cec6c95cfb7f962b6f88225b72f32209b9a))


### Bug Fixes

* **helix-picks:** carry each registration's offset from the centre line ([202ab8b](https://github.com/copick/copick-helix/commit/202ab8b3276f0d3ce722bac77ad2c8254d1af493))
* **terms:** export picks and filament lines on the recentered center line ([1010753](https://github.com/copick/copick-helix/commit/1010753602b488d9dabcb013491ab209c716d9f3))
* **terms:** write the refined axis as the filament line, and put dense picks on it ([d8e7cd3](https://github.com/copick/copick-helix/commit/d8e7cd31be1b13c46f547c778cc5da8f63071bd6))


### Performance Improvements

* registration-only iterative run when the lattice is not detected ([49417de](https://github.com/copick/copick-helix/commit/49417de3a65db1ddd9c7a774754903393566ea3a))
* straighten one run per worker (--workers) ([ca2d212](https://github.com/copick/copick-helix/commit/ca2d2126b34d2705649d55976c982c8d9c81a08a))


### Documentation

* how it works, with a flow diagram ([f1d9ce8](https://github.com/copick/copick-helix/commit/f1d9ce87ebc0131cb5f4d9187aae6238a8cd3fde))
* IF through the term route from the 10521 tilt series: no lattice ([43b4f43](https://github.com/copick/copick-helix/commit/43b4f438148f9469cb5036561149e37786199948))
* plan decisions (outputs, copick lines, conventions, open items) ([e281027](https://github.com/copick/copick-helix/commit/e281027aa3d7713fa7473b5e7dda47a6923f05fa))


### Styles

* American English throughout ([4238f4e](https://github.com/copick/copick-helix/commit/4238f4e022810990d2762c45aa24ab4f3b42c6cd))
* black and ruff ([682c106](https://github.com/copick/copick-helix/commit/682c10653fd5a3a5c2928e36901cc6e7cb78408c))


### Miscellaneous Chores

* name the polarity read-out 'invariants', README, quiet empty-slice warnings ([5d07471](https://github.com/copick/copick-helix/commit/5d07471796299f0971d90e10bb91cd145d327166))


### Build System

* copick 1 on main, gemmi and zarr-particle-tools as regular dependencies ([dc12822](https://github.com/copick/copick-helix/commit/dc128229f79fe69e5610d355deedf885a64426d1))


### Continuous Integration

* release-please, dependabot, linting and PR-title checks ([2401d0f](https://github.com/copick/copick-helix/commit/2401d0f7c05e112aabfc6a0634a46ab76cb73fd5))
