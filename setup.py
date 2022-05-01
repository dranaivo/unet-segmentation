import os
from setuptools import setup, find_packages
from typing import List

with open("README.md", "r", encoding="utf-8") as f:
    readme = f.read()


def _load_requirements(path_dir: str,
                       file_name: str = 'requirements.txt',
                       comment_char: str = '#') -> List[str]:
    """Load requirements from a file."""
    with open(os.path.join(path_dir, file_name), 'r') as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[:ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith('http'):
            continue
        if ln:    # if requirement is not empty
            reqs.append(ln)
    return reqs


setup(name='unet-segmentation',
      version='0.1.0',
      description='UNet segmentation on a cityscape-like dataset.',
      long_description=readme,
      long_description_content_type="text/markdown",
      url='https://github.com/dranaivo/unet-segmentation.git',
      install_requires=_load_requirements(path_dir=".",
                                          file_name="requirements.txt"),
      extras_require={
          'dev':
              _load_requirements(path_dir=".", file_name="dev-requirements.txt")
      },
      packages=find_packages(exclude=('tests')))
