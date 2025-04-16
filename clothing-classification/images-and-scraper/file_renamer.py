import os
import re


def rename_files(directory):
    global_counter = 1000

    # First, collect all files to rename
    files_to_rename = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if re.match(r'image_\d+\.(jpg|jpeg)$', filename):
                files_to_rename.append((root, filename))

    # Sort them to ensure consistent ordering
    files_to_rename.sort()

    # Now rename all files using the global counter
    for root, filename in files_to_rename:
        # Get the extension
        extension = filename.split('.')[-1]

        # Create new filename with global counter
        new_filename = f'image_{global_counter}.{extension}'

        # Full paths
        old_path = os.path.join(root, filename)
        new_path = os.path.join(root, new_filename)

        # Rename the file
        try:
            os.rename(old_path, new_path)
            print(f'Renamed: {filename} → {new_filename}')
            global_counter += 1
        except Exception as e:
            print(f'Error renaming {filename}: {e}')


# Use the script
directory = 'clothing_images'  # Replace with your directory path
rename_files(directory)