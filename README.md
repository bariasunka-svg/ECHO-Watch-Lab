# ECHO Watch Lab

I'm building ECHO Watch to help me monitor physical items on a display: discover what's in view, choose what matters, remember those items, and review what happened when they move, disappear or return.

My goal is a clear desktop workbench where I can see the scene, follow individual items and investigate events without piecing together separate tools. I'm developing it as a native Windows application with Python and Tkinter.

## Where I am now

I have a working local-video detection and tracking workspace, and I've started testing it manually. I can choose a video, run YOLOE vocabulary detection, see bounding boxes and BoT-SORT track IDs, pause or stop playback, and open the saved test results.

I also have a Display Assets workflow for discovering items, selecting references, configuring monitoring and reviewing saved runs. I'm still evaluating its recognition and remembered-item behavior. I consider this an experimental development build, with more work needed on accuracy, consistency and usability.

My current source is **prerecorded local video**. Live camera support is on my roadmap.

## What I'm testing

### YOLOE workspace

I use this workspace to understand the detector and tracker before adding more item-memory behavior.

1. Choose a local video and press **Start**.
2. Watch vocabulary labels, confidence scores and track IDs on the analysed images.
3. Switch between **Detections + tracks** and **Detections only**, or adjust which confidence scores are shown.
4. Select a box or detection row to inspect it.
5. Pause, resume, stop or let the video finish.
6. Open **View test results** to inspect the saved records.

I keep detections visible even when the tracker has not assigned an ID. I treat a track ID as continuity within a test, rather than proof that an object is the same physical item across separate visits or runs.

### Display Assets

I'm developing the main monitoring journey around four steps:

**Discover → Select → Configure → Monitor**

I want to discover visible items, choose the ones I care about, save their references, and monitor them with a paired scene view and item tray. I keep saved Library profiles separate from the items selected for the current run. Stopping a run clears its current membership while preserving the Library and saved history.

The workbench includes recording, activity, decision traces, frame investigation and historical corrections with Undo. I'm still refining how these pieces fit together in everyday use.

## My next steps

1. Test a representative set of videos and improve detection and tracking based on repeatable failures.
2. Add live camera input, including clear handling of delays and disconnects.
3. Repeat the same scenarios with a real camera.
4. Develop remembered-item identity: select a particular item and assess whether that same item returns.
5. Refine the whole experience, with usability improvements throughout development.

I don't expect every object to be detected or named correctly. Small objects, clutter, poor lighting, occlusion and similar-looking items are part of what I need to test. I want uncertainty to remain visible whenever the evidence is weak.

## Technical approach

I keep the interface and inference in separate project-local Python runtimes. The Tkinter application manages the session and presentation; an isolated worker runs the pinned YOLOE model on CUDA in FP32. I currently use BoT-SORT with appearance re-identification disabled.

I pair each displayed result with the exact image that was analysed. I don't carry an old box onto newer pixels. For video evaluation, I process frames in order and allow playback to slow down when analysis cannot keep up. That makes the current replay mode useful for inspection; it does not establish live-camera performance.

Only one analysis workspace runs at a time. I save test metadata and frame-linked detection records so I can investigate failures after a run.

## Running my development setup

I currently run this on Windows with project-local Python 3.13 runtimes and an NVIDIA CUDA-capable GPU. This repository contains the source, tests, dependency locks and development tools. I keep model checkpoints, Python environments, native binaries, videos and operator data outside Git.

My local layout is:

```text
env/ui/       Python with Tkinter and the UI dependencies
env/gpu/      Python with the pinned inference dependencies
env/dev-site/ Test dependencies
models/       The pinned YOLOE prompt-free and visual-prompt checkpoints
```

I record dependency versions and artifact hashes in [requirements](requirements/) and [config.py](src/echo_watch/config.py). Each runtime needs the project's `src` directory on its import path. My isolated Python setup uses its local `python313._pth` for that configuration.

Once those local dependencies and models are provisioned, I launch the workbench from the repository root:

```powershell
.\env\ui\pythonw.exe -I -m echo_watch
```

I'm not presenting this snapshot as a one-command installation or a packaged release. Some bootstrap and qualification tools still refer to my local development archives and paths.

## Repository layout

| Path | What I keep there |
| --- | --- |
| `src/echo_watch/app/` | Desktop interface, controllers and display preparation |
| `src/echo_watch/yoloe/` | Detection/tracking workspace, worker, ordered replay and saved test review |
| `src/echo_watch/engine/` | Monitoring sessions, Library, recording and saved setups |
| `src/echo_watch/review/` | Saved-run investigation, activity and corrections |
| `tests/` | Focused regression tests and synthetic test fixtures |
| `requirements/` | Pinned dependency locks |
| `tools/` | Development, provisioning and qualification tools |
| `docs/SOURCE_PROVENANCE.json` | Source adaptation records |
| `docs/donor-notices/` | Preserved attribution and license notices |

## How I judge progress

I use regression tests to check behavior, and I manually test the rendered application. I keep those results separate: passing tests and high inference throughput don't establish that the interface is usable or that detection is accurate on a new video.

My next evaluation focuses on missed objects, incorrect labels, inaccurate boxes, ID switches, removal and return, and whether playback and controls remain understandable.

## License and attribution

I publish the project's own source under **GNU AGPL-3.0-only**. See [LICENSE](LICENSE), [NOTICE.md](NOTICE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

I preserve the attribution for adapted source in [the provenance record](docs/SOURCE_PROVENANCE.json) and [donor notices](docs/donor-notices/). Third-party dependencies, model weights and media retain their respective terms; I don't bundle those artifacts in this repository.
