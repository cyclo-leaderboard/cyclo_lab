# WallRoom

BG2 1063 table workspace shared by Task000400 and Task000853. The scene is a
standalone USD, with no WhiteRoom or showroom scene sublayer. It has a colliding
floor, a 0.735 m table, one continuous rear wall, and one solid wall on each
camera side. Wall materials are fixed to the BG2 reference view; they are not
calibrated photometric measurements.

`wall_room_lighting.usda` contains two lights on the robot/table side of the rear
wall. Task000853 overrides only the camera-right wall position in its own layer.
The wall has the same visual and collision geometry. The separate `work_table`
asset and texture are included here so the scene does not depend on the ignored
WhiteRoom package at runtime.

The Python factory is `cyclo_lab.assets.environments.wall_room`. The task
plugins use `Cyclo-Real-WallRoom-Task000400-FFW-BG2-v0` and
`Cyclo-Real-WallRoom-Task000853-FFW-BG2-v0`. Historical WhiteRoom Gym IDs remain
read aliases for datasets recorded before this change.

USD composition and rendered/physics behavior still require an Isaac Sim check
before treating this geometry as a validated real-world match.
