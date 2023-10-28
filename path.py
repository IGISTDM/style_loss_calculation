import os
import sys

def path_join(*path_components):
    # Join the path components using os.path.join()
    path_with_backslashes = os.path.join(*path_components)

    if sys.platform.startswith('win'):
        # Replace backslashes with forward slashes
        path_with_forwardslashes = path_with_backslashes.replace('\\', '/')

    return path_with_forwardslashes


