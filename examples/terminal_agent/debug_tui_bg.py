from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, Window, HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.utils import get_cwidth
import shutil

def get_text():
    width = shutil.get_terminal_size().columns
    lines = []
    
    # Test line 1: Short text, manual padding
    text = "Short text"
    padding = " " * (width - len(text))
    lines.append([("bg:green", text + padding)])
    
    # Test line 2: Prefix + Content + Padding
    prefix = "1234 "
    content = "Some content"
    current_len = get_cwidth(prefix) + get_cwidth(content)
    padding = " " * (width - current_len)
    lines.append([
        ("bg:red", prefix),
        ("bg:red", content + padding)
    ])
    
    # Flatten
    result = []
    for l in lines:
        result.extend(l)
        result.append(("", "\n"))
    return result

layout = Layout(HSplit([
    Window(FormattedTextControl(get_text), height=10),
    Window(height=1, char="-", style="bg:blue"),
    Window(FormattedTextControl("Press q to quit"), height=1)
]))

kb = KeyBindings()
@kb.add('q')
def _(event):
    event.app.exit()

app = Application(layout=layout, key_bindings=kb, full_screen=True)
app.run()
