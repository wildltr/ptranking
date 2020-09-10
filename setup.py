import setuptools

long_description = ''

install_requires = [
    'numpy',
    'scikit-learn',
    'tqdm',
    #'torch >= 1.6.0', # todo torch-gpu
    #'torchvision',
]

extras_requires = None

setuptools.setup(
    name="ptranking",
    version="0.8.1",
    author="II-Research",
    author_email="yuhaitao@slis.tsukuba.ac.jp",
    description="A library of scalable and extendable implementations of typical learning-to-rank methods based on PyTorch.",
    license="MIT License",
    keywords=['Learning-to-rank', 'PyTorch'],
    url="https://ptranking.github.io",
    packages=setuptools.find_namespace_packages(include=["ptranking", "ptranking.*"]),
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=install_requires,
    extras_require=extras_requires
)

#todo package_data
