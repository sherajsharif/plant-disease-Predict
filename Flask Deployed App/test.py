import os

UPLOAD_FOLDER = os.path.join('static', 'uploads')

# Check if the directory exists
if os.path.exists(UPLOAD_FOLDER):
    print("✅ The folder 'static/uploads' exists.")
else:
    print("❌ The folder 'static/uploads' does NOT exist. Creating it now...")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Verify again
if os.path.exists(UPLOAD_FOLDER):
    print("✅ The folder is now created successfully!")
else:
    print("❌ Something went wrong!")
