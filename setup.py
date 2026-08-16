# Import required functions from setuptools
# find_packages() automatically finds all Python packages in the project
# setup() is used to configure and package the project
from setuptools import find_packages, setup


# Define the project/package configuration
setup(
    # Name of the project/package
    name="mediguide",

    # Current version of the project
    version="0.1.0",

    # Project authors with student IDs
    author="Saurabh Kumbhar (25204974), Azim Hassan (25203062)",

    # Short description of the MediGuide project
    description="MediGuide - AI-Powered Analysis for Intelligent Healthcare Assistance",

    # Automatically discover all Python packages
    # inside the project directory
    packages=find_packages(),

    # Dependencies are managed separately in requirements.txt
    install_requires=[]
)