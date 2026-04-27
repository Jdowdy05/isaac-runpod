from setuptools import find_packages, setup


setup(
    name="op3_teleop_lab",
    version="0.1.0",
    description="Isaac Lab external project for OP3 teleoperation locomotion",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.24",
        "torch>=2.2",
        "gymnasium>=0.29",
        "pyyaml>=6.0",
        "smplx>=0.1.28",
    ],
    package_data={
        "op3_teleop_lab.tasks.direct.g1_teleop.agents": ["*.yaml"],
        "op3_teleop_lab.tasks.direct.op3_teleop.agents": ["*.yaml"],
    },
)
