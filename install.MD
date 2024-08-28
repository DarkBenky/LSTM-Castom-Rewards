# Create a virtual environment (optional but recommended)
python -m venv tf_agents_env
source tf_agents_env/bin/activate  # On Windows use `tf_agents_env\Scripts\activate`

# Install TensorFlow
pip install tensorflow==2.15.0

# Install dm-reverb
pip install dm-reverb==0.13.0

# Install TF-Agents with Reverb support
pip install tf-agents[reverb]==0.18.0

# Install TensorFlow Probability (often needed with TF-Agents)
pip install tensorflow-probability

# Install additional useful packages
pip install numpy matplotlib

# Verify installations
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import tf_agents; print(tf_agents.__version__)"
python -c "import reverb; print(reverb.__version__)"