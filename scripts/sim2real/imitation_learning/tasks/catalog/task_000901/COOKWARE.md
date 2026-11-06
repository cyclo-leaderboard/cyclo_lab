# Measured cookware, 2026-11-02

`scene.py` includes the video-reference pot and separate glass lid alongside
the cucumber and onion. Pot dimensions supplied by the user: outer body
diameter 200 mm, body height 80 mm, total handle span 340 mm.

Assets: `runtime/data/objects/cookware_v1/{pot/pot.usd,lid/lid.usd}`.
Keep their PNG files beside the USDs. These are static visual assets; no
grasp/contact model is configured. Lid diameter and unseen geometry are
estimated; this is reference-guided modeling rather than a complete scan.

Scene snapshot: `runtime/data/task000901_with_cookware.usda`.
Choose `/World/CookwarePreviewCamera`. The normal registered Task 901 also
spawns the cookware on its next launch. Existing sessions require a reload.

Glass uses OmniGlass with IOR 1.491, translucency enabled, and RTX's
`primvars:enableFastRefractionShadow` for an approximate transmissive shadow.
GLB transparency is approximate; use USD for Isaac rendering.

Source, scripts, renders and validation:
`/home/robotis-ai/agents/experiments/2026-11-03_09-56-43_task000901_vision/cookware/`.
The scene was rendered through the actual catalog registration. No hardware
commands were sent. Robot pose and camera configuration were preserved.
