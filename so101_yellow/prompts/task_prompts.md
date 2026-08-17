# Task prompt variation matrix

Each row is one call to `scripts/06_record_dataset.sh "<prompt>" <episodes>`. All
rows accumulate into the same `DATASET_REPO_ID`. LeRobot's own guidance: aim
for ~10 episodes per variation, ≥50 episodes total before fine-tuning, and
avoid stacking too many variations into a single early recording pass (color
+ object + grasp-style all changing at once makes it harder for SmolVLA to
learn what actually matters).

Write prompts in the same style Red/Blue's corrections will eventually
arrive in, since those are the natural-language inputs the fine-tuned policy
needs to generalize to (e.g. "don't pick up the green block, pick the red
one" resolves down to a single-object instruction like row 1 below; "don't
pick it up like that" resolves to a grasp-style correction like row 4).

| # | Object color | Object type | Grasp style         | Prompt string                                        | Episodes |
|---|---------------|-------------|----------------------|-------------------------------------------------------|----------|
| 1 | red           | cube        | top-down pinch       | "Pick up the red cube and place it in the bin"         | 10       |
| 2 | blue          | cube        | top-down pinch       | "Pick up the blue cube and place it in the bin"        | 10       |
| 3 | green         | cylinder    | side grasp           | "Pick up the green cylinder and place it in the bin"   | 10       |
| 4 | red           | cube        | side grasp (retry)   | "Pick up the red cube using a side grasp"              | 10       |
| 5 | yellow        | sphere      | top-down pinch       | "Pick up the yellow ball and place it in the bin"      | 10       |

Add rows as needed. Keep camera position/lighting fixed across a recording
session so variation in the data comes from the task, not the setup.
