# LACE Demo

## Installation

```shell
# Create a virtualenv folder
mkdir -p ~/venv-environments/lace

# Create a new virtualenv in that folder
python3 -m venv ~/venv-environments/lace

# Activate the virtualenv
source ~/venv-environments/lace/bin/activate

# Install deps
pip install flask numpy pandas scipy snapshottest scikit-learn matplotlib seaborn 
pip install --no-binary orange3 orange3==3.15.0
```

## Running the demo

### Running the backend
```shell
FLASK_ENV=development flask run
```

### Running the frontend
```shell
cd demo_frontend/my-app
npm install # Only needed on the first run
npm start
```
Visit `localhost:3000` in a browser.
# X-PLAIN_Demo
