from setuptools import setup, find_packages

install_requires = [
    'certifi >= 2021.5.30',
    'charset-normalizer >= 2.0.6',
    'colorama >= 0.4.4',
    'cycler >= 0.10.0',
    'flatbuffers >= 2.0',
    'idna >= 3.2',
    'kiwisolver >= 1.3.2',
    'matplotlib >= 3.4.3',
    'numpy >= 1.21.2',
    'onnx >= 1.10.1',
    'onnxruntime >= 1.9.0',
    'opencv-python >= 4.1.2',
    'Pillow >= 8.3.2',
    'protobuf >= 3.18.1',
    'pydot >= 1.4.2',
    'pyparsing >= 2.4.7',
    'python-dateutil >= 2.8.2',
    'requests >= 2.26.0',
    'six >= 1.16.0',
    'torch >= 1.7',
    'torchvision >= 0.8.0',
    'torchinfo >= 1.5.3',
    'tqdm >= 4.62.3',
    'typing-extensions >= 3.10.0.2',
    'urllib3 >= 1.26.7',
]

setup(
    name='dxeon',
    version='0.1.0',

    python_requires='>=3.6',

    packages=find_packages(),
    include_package_data=True,

    author='Rishik C. Mourya',
    author_email='braindotai@gmail.com',
    description='Fast task utils for deep learning and machine learning pipelines.',
    url='https://github.com/braindotai/Dxeon',
    install_requires=install_requires,
    license='MIT',
)