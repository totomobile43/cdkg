from setuptools import setup

setup(name='CDKG',
      description='Create designs for motion resisting garments automatically using topology optimization.',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      url="https://github.com/totomobile43/cdkg",
      version='1.0.0',
      author='Velko Vechev',
      author_email="velko.vechev@inf.ethz.ch",
      packages=['cdkg'],
      include_package_data=True,
      keywords=['computational tool', 'optimization'],
      platforms=['any'],
      python_requires='>=3.7',
      install_requires=[
            'aitviewer>=1.11',
            'pytorch3d>=0.4.0',
            'plotly'
          ],
      )
