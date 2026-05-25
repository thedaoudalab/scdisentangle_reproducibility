import os

def log(log_message: str):
    print(f'{log_message}\n')

def add_sep(path: str, make_dir: bool = True) -> str:
    """ Add os separator to a path """
    if not isinstance(path, str):
        log('Please provie a valid save path')
        return
    if not path.endswith(os.sep):
        path += os.sep
    # Create folder if not exists:
    if not os.path.exists(path) and make_dir:
        log(f'Creating the path {path}\n')
        os.mkdir(path)
    return path

def _make_tree(path):
    folders = path.split('/')
    progressive_path = ''
    for folder in folders:
        progressive_path += folder + '/'
        if folder != '':
            _make_dir(progressive_path)

# def _make_tree(*folders):
#     """ Make tree of folders """
#     current_dir = add_sep('experiments/', True)
#     for folder in folders:
#         folder = folder.replace('.', '_')
#         current_dir += add_sep(folder, make_dir = False)
#         _make_dir(current_dir)
#     return '/'.join(current_dir.split('/')[1:])

def _make_dir(path):
    """ Make dir if not exists """
    if not os.path.exists(path):
        os.mkdir(path)