# [2.4.0](https://github.com/kevinriste/podly_pure_podcasts/compare/v2.3.0...v2.4.0) (2026-03-07)


### Bug Fixes

* **ci:** add source path to mypy install-types and restore mypy cache ([0ae6b65](https://github.com/kevinriste/podly_pure_podcasts/commit/0ae6b6582079fc6057a889b0c76aee7996e0d5be))
* **ci:** collapse redundant mypy steps into one ([f71dfdf](https://github.com/kevinriste/podly_pure_podcasts/commit/f71dfdf780f5d0d9f08898ef28d198c954cefbd8))
* **ci:** remove --install-types from mypy invocations ([b6e97da](https://github.com/kevinriste/podly_pure_podcasts/commit/b6e97da8f9cf85074485199bfd704cd5ee58cfba))
* **ci:** restore optional integration check flag ([1dbfb5a](https://github.com/kevinriste/podly_pure_podcasts/commit/1dbfb5aea96c72d3ae772e5400cfcc4c49ac5fb6))
* **docker:** avoid undocumented --project file path in uv sync ([7281803](https://github.com/kevinriste/podly_pure_podcasts/commit/728180359a9803e5c97c9f6590a9075b55487cfb))
* **docker:** install deps via uv export requirements ([a2aba0e](https://github.com/kevinriste/podly_pure_podcasts/commit/a2aba0e3f32af09817ad9dcab671b8a96893cbfd))
* **docker:** pin uv version and remove dead pip env vars ([58135fb](https://github.com/kevinriste/podly_pure_podcasts/commit/58135fb5e61ffc23722bef7711a593abef7a1ed4))
* **docker:** restore diagnostic echo lines in dependency install ([c531dd1](https://github.com/kevinriste/podly_pure_podcasts/commit/c531dd1461d1cc5e56536f578edc148835a39827))
* **docker:** use uv sync and uv run in container ([e1a752d](https://github.com/kevinriste/podly_pure_podcasts/commit/e1a752db6959dfe8ae7ce43a20c9988e0c68742d))
* double casting ([191612f](https://github.com/kevinriste/podly_pure_podcasts/commit/191612f3580eb8c562f6683cbc4ee474d6e2cab8))
* enhance word boundary refiner with improved JSON output handling. Output was geting truncated resulting in wasted LLM calls. ([c66f139](https://github.com/kevinriste/podly_pure_podcasts/commit/c66f1392c90649149eb5bc8351eb922f1abf8578))
* Implement auto-whitelist check for first member in feed ([c052e93](https://github.com/kevinriste/podly_pure_podcasts/commit/c052e93752a6301b33d13e270aa46b4e53b4bf7b))
* Implement auto-whitelist check for first member in feed so ([38d320a](https://github.com/kevinriste/podly_pure_podcasts/commit/38d320a2332257f8e2252a55e4f730ab5f0326b1))
* Implement auto-whitelist check for first member in feed so ([ad3fe13](https://github.com/kevinriste/podly_pure_podcasts/commit/ad3fe13b90db423a9bac00f0da1354803faf559d))
* **jobs:** skip job creation for whitelisted posts with existing processed audio ([1172f02](https://github.com/kevinriste/podly_pure_podcasts/commit/1172f0243a2a3e6b9383ab7a7ec5fe9073111cbc))
* **scripts:** restore --install-types --non-interactive to ci.sh mypy ([c6a9853](https://github.com/kevinriste/podly_pure_podcasts/commit/c6a9853729faa64ea104fb98d00f25ddbfa15f2d))


### Features

* add advertisement removal summary and processing stats to the UI ([046d112](https://github.com/kevinriste/podly_pure_podcasts/commit/046d112b08fafea629472f536e3a2ea378c3b234))
* Add advertisement removal summary and timeline visualization to LLMProcessingStats component that was accidently removed in a previous refactoring  PR ([63682d9](https://github.com/kevinriste/podly_pure_podcasts/commit/63682d9ad21f63435d6635a3fd07f8a5e746b0de))
* add build options for AMD64 and ARM64 images in Docker workflow ([af5eb48](https://github.com/kevinriste/podly_pure_podcasts/commit/af5eb484ded7ac5c95f9f677bf2281fe113841ae))
* add debug information for post stats including audio paths and processing details ([6fcc0bc](https://github.com/kevinriste/podly_pure_podcasts/commit/6fcc0bc2089d8dc5221c9cf122045f515ca5a930))
* Add feed detail route and handle feed selection in HomePage component ([a4d8d7e](https://github.com/kevinriste/podly_pure_podcasts/commit/a4d8d7eb616f186a93de9d74a4429f9a11a2c2e9))
* Add feed detail route and handle feed selection in HomePage component ([8a81f5c](https://github.com/kevinriste/podly_pure_podcasts/commit/8a81f5c767416967342610405887288b3b41ffa5))
* add functionality to cancel all queued jobs with API integration and UI support ([0387ec9](https://github.com/kevinriste/podly_pure_podcasts/commit/0387ec92b82822d1a0e04a1882b9f14fceb03dfc))
* add manual workflow inputs for Docker image builds and update conditions for job execution ([6d448d1](https://github.com/kevinriste/podly_pure_podcasts/commit/6d448d112d6a866a452d78dc26eac93277221768))
* add support for manual workflow dispatch in Docker publish process ([40abc03](https://github.com/kevinriste/podly_pure_podcasts/commit/40abc03bad25234fdf97c58b7978e66349dbbd09))
* Enhance episode description handling with expandable view and HTML decoding ([5fa0c77](https://github.com/kevinriste/podly_pure_podcasts/commit/5fa0c7752590f6f84204c1ee4f872814222470be))
* Implement advanced episode description parsing and rendering ([0736bfe](https://github.com/kevinriste/podly_pure_podcasts/commit/0736bfeabe7c645df0bafe5460ec02ae3b64414a))
* implement audio path resolution in JobManager and add unit tests for validation logic ([6318fd1](https://github.com/kevinriste/podly_pure_podcasts/commit/6318fd1026a4f7c0f0c5eef91f10aa2c8de39bd4))
* implement find_existing_processed_audio_path utility and refactor audio path handling in jobs manager and cleanup actions ([c96f183](https://github.com/kevinriste/podly_pure_podcasts/commit/c96f183c007d7d2730dea5f21609d47fc33f7c54))
* retain more info on cleanup and integration check ([80069fc](https://github.com/kevinriste/podly_pure_podcasts/commit/80069fcf0f1119e4201417b9f372c6a2730ca62d))
* retain most recent post after cleanup ([add27e1](https://github.com/kevinriste/podly_pure_podcasts/commit/add27e143519942c548221d378dc49c99c0b7472))
* simplify conditional checks for Docker build jobs and add variant selection logic ([c66a2ec](https://github.com/kevinriste/podly_pure_podcasts/commit/c66a2eccd286b8a4f92bf10f4c7e6760f8d775e0))
