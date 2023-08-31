import shutil

term_width, _ = shutil.get_terminal_size()
help_width = min(term_width, 100)
CLICK_SETTINGS = dict(help_option_names=["-h", "--help"], max_content_width=help_width)