from setuptools import find_packages, setup


with open('requirements.txt') as f:
    required = f.read().splitlines()


setup(
    name='model',
    packages=find_packages(),
    version='0.1.0',
    description='HW1',
    author='Vladimir Shaposhnikov',
    author_email='shaposh-vova@rambler.ru',
    entry_points={
        'console_scripts': [
            'ml_example_train = model.train:train_model'
        ]
    },
    install_requires = required
)