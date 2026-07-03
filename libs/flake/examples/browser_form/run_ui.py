import time
from pathlib import Path
from bench_ui import launch_window

HTML = Path(__file__).with_name("task.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    launch_window(html=HTML, title="Form Task", x=0, y=0, width=800, height=800)
    # Keep process alive so driver can interact with it
    while True:
        time.sleep(1)
