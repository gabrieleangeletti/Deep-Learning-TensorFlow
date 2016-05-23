from setuptools import setup, find_packages

setup(
    name='yadlt',
    version='0.0.2rc-1',
    url='https://github.com/blackecho/Deep-Learning-Tensorflow',
    download_url='https://github.com/blackecho/Deep-Learning-TensorFlow/tarball/0.0.2rc-1',
    author='Gabriele Angeletti',
    author_email='angeletti.gabriele@gmail.com',
    description='Implementation of various deep learning algorithms using Tensorflow. Class interfaces is sklearn-like.',
    packages=find_packages(exclude=['tests', 'data', 'logs', 'stored_models']),
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    license='MIT',
    install_requires=[],
)
