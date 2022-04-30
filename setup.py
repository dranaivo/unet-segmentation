from setuptools import setup, find_packages


with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()

setup(
    name='unet-segmentation',
    version='0.1.0',
    description='UNet segmentation on a cityscape-like dataset.',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/dranaivo/unet-segmentation.git',
    packages=find_packages(exclude=('tests'))
)

