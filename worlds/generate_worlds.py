"""
Generates all evaluation-scenario Gazebo world files from a template.

CHANGE vs. original: accepts --template so the SAME obstacle layouts can be
built either against _base_template.world (real_time_factor=1, for the
live VNC demo / video recording) or _base_template_fast.world
(real_time_update_rate=0, unthrottled, for training). Output filenames get
a suffix so both sets can coexist:

    python3 generate_worlds.py --template fast     # -> static_obstacles_fast.world, etc.
    python3 generate_worlds.py --template realtime # -> static_obstacles.world, etc. (default)

Scenarios (per assessment brief, section 04):
  1. open              -- no obstacles (uses the existing rl_empty_world.world, skipped here)
  2. static_obstacles   -- several fixed boxes/cylinders between plausible start/goal areas
  3. unseen             -- a different layout never used during training, for generalization testing
  4. narrow_passage     -- two obstacles close enough together to require careful maneuvering
  5. dead_end           -- a U-shaped wall the robot must recognize and back out of
  6. dynamic_obstacle    -- static_obstacles layout + one movable cylinder driven by
                            evaluate.py's --dynamic-obstacle mode via /gazebo/set_entity_state
"""
import argparse
import os

HERE = os.path.dirname(__file__)


def box(name, x, y, z=0.25, sx=0.4, sy=0.4, sz=0.5):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
          <material><ambient>0.6 0.2 0.2 1</ambient></material>
        </visual>
      </link>
    </model>"""


def wall_segment(name, x, y, length, yaw, thickness=0.1, height=0.5):
    return f"""
    <model name="{name}">
      <static>true</static>
      <pose>{x} {y} {height/2} 0 0 {yaw}</pose>
      <link name="link">
        <collision name="collision">
          <geometry><box><size>{length} {thickness} {height}</size></box></geometry>
        </collision>
        <visual name="visual">
          <geometry><box><size>{length} {thickness} {height}</size></box></geometry>
          <material><ambient>0.3 0.3 0.6 1</ambient></material>
        </visual>
      </link>
    </model>"""


def dynamic_box(name, x, y, amplitude=1.0, axis="y", speed=0.3, z=0.25):
    # Placed here as a kinematic body; evaluate.py's --dynamic-obstacle mode
    # drives it via /gazebo/set_entity_state during each episode (documented
    # limitation, not a hidden hack -- Gazebo Classic has no simple SDF-only
    # scripted-motion plugin for this).
    return f"""
    <model name="{name}">
      <static>false</static>
      <pose>{x} {y} {z} 0 0 0</pose>
      <link name="link">
        <collision name="collision">
          <geometry><cylinder><radius>0.15</radius><length>0.5</length></cylinder></geometry>
        </collision>
        <visual name="visual">
          <geometry><cylinder><radius>0.15</radius><length>0.5</length></cylinder></geometry>
          <material><ambient>0.6 0.5 0.1 1</ambient></material>
        </visual>
        <inertial><mass>1.0</mass></inertial>
      </link>
    </model>"""


def build_world(template_path, filename, obstacles_xml):
    with open(template_path) as f:
        tmpl = f.read()
    out = tmpl.replace("<!-- OBSTACLES_PLACEHOLDER -->", obstacles_xml)
    out_path = os.path.join(HERE, filename)
    with open(out_path, "w") as f:
        f.write(out)
    print(f"wrote {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", choices=["fast", "realtime"], default="realtime",
                         help="'fast' = unthrottled physics for training; "
                              "'realtime' = real_time_factor=1 for VNC demo/video recording")
    args = parser.parse_args()

    if args.template == "fast":
        template_path = os.path.join(HERE, "_base_template_fast.world")
        suffix = "_fast"
    else:
        template_path = os.path.join(HERE, "_base_template.world")
        suffix = ""

    static_obs = "".join([
        box("obs1", 1.0, 0.5),
        box("obs2", -0.8, 1.2),
        box("obs3", 1.5, -1.0),
        box("obs4", -1.5, -0.8),
    ])
    build_world(template_path, f"static_obstacles{suffix}.world", static_obs)

    unseen_obs = "".join([
        box("u_obs1", 0.6, -1.3, sx=0.5, sy=0.3),
        box("u_obs2", -1.2, 0.3, sx=0.3, sy=0.6),
        box("u_obs3", 2.0, 1.5, sx=0.4, sy=0.4),
        wall_segment("u_wall1", 0.0, 2.0, 2.0, 0.0),
    ])
    build_world(template_path, f"unseen{suffix}.world", unseen_obs)

    narrow_obs = "".join([
        box("n_obs_left", 0.0, 0.28, sx=0.3, sy=0.5, sz=0.5),
        box("n_obs_right", 0.0, -0.28, sx=0.3, sy=0.5, sz=0.5),
    ])
    build_world(template_path, f"narrow_passage{suffix}.world", narrow_obs)

    dead_end_obs = "".join([
        wall_segment("de_back", 1.5, 0.0, 1.5, 1.5708),
        wall_segment("de_left", 1.0, 0.7, 1.0, 0.0),
        wall_segment("de_right", 1.0, -0.7, 1.0, 0.0),
    ])
    build_world(template_path, f"dead_end{suffix}.world", dead_end_obs)

    dynamic_obs = static_obs + dynamic_box("moving_obstacle", 0.5, 0.0)
    build_world(template_path, f"dynamic_obstacle{suffix}.world", dynamic_obs)


if __name__ == "__main__":
    main()
