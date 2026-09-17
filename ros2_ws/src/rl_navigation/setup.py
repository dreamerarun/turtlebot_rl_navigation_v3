import os
from glob import glob
from setuptools import find_packages, setup

package_name = "rl_navigation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Arun",
    maintainer_email="arunmurugesan0110@gmail.com",
    description="RL policy deployment node for TurtleBot3 navigation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "policy_node = rl_navigation.policy_node:main",
        ],
    },
)
