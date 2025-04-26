#!/bin/bash

# Configure Git user information
read -p "Enter your Git username: " username
read -p "Enter your Git email: " email

git config --global user.name "$username"
git config --global user.email "$email"

# Initialize git repository
git init

# Create .gitignore file with common Python patterns
cat > .gitignore << EOL
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb

# Virtual Environment
venv/
env/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
EOL

# Add all files to git
git add .

# Make initial commit
git commit -m "Initial commit"

echo "Git repository initialized successfully!"
echo "Git user configured as: $username <$email>" 