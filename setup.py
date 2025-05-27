from setuptools import setup, find_packages

setup(
    name='tsanalyzer',
    version='0.1.0',
    packages=find_packages(),
    author='Zhenjiang Wu',
    author_email='zhenjiangwu@stu.xjtu.edu.cn',
    description='Analyze time series for periodicity, randomness, and structural changes.',
    url='https://github.com/JustinLo-ops/tsanalyzer.git',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.6',
)
