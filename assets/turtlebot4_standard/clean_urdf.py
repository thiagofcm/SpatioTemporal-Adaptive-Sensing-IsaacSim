"""Clean the xacro-expanded TurtleBot4 Standard URDF for the Isaac Sim URDF importer.

- removes <gazebo>, <ros2_control> and <transmission> elements (simulator plugins, not geometry/physics)
- resolves package://<pkg>/... mesh URIs to absolute paths in the pinned clones (references/robot_description/)
- prints a summary of links (mass, collision geometry) and movable joints (limits, dynamics)

Usage: python3 clean_urdf.py <raw.urdf> <clean.urdf>
"""
import os
import re
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
TP = os.path.normpath(os.path.join(HERE, "..", "..", "references", "robot_description"))
PKG_DIRS = {}
for repo in ("turtlebot4", "create3_sim"):
    for d, _, files in os.walk(os.path.join(TP, repo)):
        if "package.xml" in files:
            m = re.search(r"<name>\s*([^<\s]+)\s*</name>", open(os.path.join(d, "package.xml")).read())
            if m:
                PKG_DIRS[m.group(1)] = d

src, dst = sys.argv[1], sys.argv[2]
tree = ET.parse(src)
root = tree.getroot()
removed = {}
for tag in ("gazebo", "ros2_control", "transmission"):
    for el in root.findall(tag):
        root.remove(el)
        removed[tag] = removed.get(tag, 0) + 1
for mesh in root.iter("mesh"):
    fn = mesh.get("filename")
    m = re.match(r"package://([^/]+)/(.*)", fn)
    if m:
        path = os.path.join(PKG_DIRS[m.group(1)], m.group(2))
        assert os.path.exists(path), path
        mesh.set("filename", path)
# Wheel-drop suspension: in Gazebo a plugin provides the spring; without it the prismatic joints (damping only)
# would collapse under gravity. Fix them at their zero (nominal, on-ground) position.
fixed = []
for j in root.findall("joint"):
    if j.get("type") == "prismatic" and j.get("name").startswith("wheel_drop"):
        j.set("type", "fixed")
        for tag in ("limit", "dynamics", "axis"):
            el = j.find(tag)
            if el is not None:
                j.remove(el)
        fixed.append(j.get("name"))
print("fixed wheel-drop joints:", fixed)
# Wheel colliders: the 15 mm wide cylinders make line contacts whose inner/outer edges scrub when the robot turns
# (PhysX resolves several contact points across the width). A sphere of the same radius gives a single rolling contact
# point (ideal differential drive). Inertia and visuals are unchanged; the sphere (r = 0.03575 m at y = +-0.1165 m)
# stays inside the robot footprint (bumper r = 0.177 m). Disable with SPHERE_WHEELS=0.
if os.environ.get("SPHERE_WHEELS", "1") == "1":
    for name in ("left_wheel", "right_wheel"):
        link = root.find(f"link[@name='{name}']")
        for col in link.findall("collision"):
            geom = col.find("geometry")
            cyl = geom.find("cylinder")
            if cyl is not None:
                r = cyl.get("radius")
                geom.remove(cyl)
                ET.SubElement(geom, "sphere", {"radius": r})
                print(f"wheel collider {name}: cylinder -> sphere r={r}")
tree.write(dst, xml_declaration=True, encoding="utf-8")
print("removed:", removed)

total_mass = 0.0
print("\n# links with mass or collision")
for link in root.findall("link"):
    inert = link.find("inertial")
    mass = float(inert.find("mass").get("value")) if inert is not None else 0.0
    total_mass += mass
    cols = []
    for c in link.findall("collision"):
        g = c.find("geometry")[0]
        o = c.find("origin")
        cols.append(f"{g.tag}{dict(g.attrib) if g.tag != 'mesh' else '(' + os.path.basename(g.get('filename')) + ')'}"
                    f"@{o.get('xyz') if o is not None else '0 0 0'}")
    if mass > 0 or cols:
        print(f"  {link.get('name'):32s} mass={mass:.3f}  collisions={cols}")
print(f"  total mass = {total_mass:.3f} kg")
print("\n# movable joints")
for j in root.findall("joint"):
    if j.get("type") == "fixed":
        continue
    lim, dyn, o, ax = j.find("limit"), j.find("dynamics"), j.find("origin"), j.find("axis")
    print(f"  {j.get('name')} [{j.get('type')}] parent={j.find('parent').get('link')} child={j.find('child').get('link')}"
          f" origin={o.attrib if o is not None else None} axis={ax.get('xyz') if ax is not None else None}"
          f" limit={lim.attrib if lim is not None else None} dynamics={dyn.attrib if dyn is not None else None}")
for name in ("base_footprint_joint", "front_caster_joint", "rplidar_joint", "shell_link_joint"):
    j = root.find(f"joint[@name='{name}']")
    print(f"  {name}: parent={j.find('parent').get('link')} child={j.find('child').get('link')} origin={j.find('origin').attrib}")
