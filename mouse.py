"""Mouse control utility for interacting with the game."""
import sys
import time
import pyautogui

# Safety: don't move too fast, allow abort by moving mouse to corner
pyautogui.PAUSE = 0.3
pyautogui.FAILSAFE = True


def click(x, y, button="left"):
    """Click at position (x, y)."""
    pyautogui.click(x, y, button=button)
    print(f"Clicked ({x}, {y}) {button}")


def move(x, y):
    """Move mouse to position (x, y)."""
    pyautogui.moveTo(x, y)
    print(f"Moved to ({x}, {y})")


def right_click(x, y):
    """Right-click at position (x, y)."""
    pyautogui.click(x, y, button="right")
    print(f"Right-clicked ({x}, {y})")


def double_click(x, y):
    """Double-click at position (x, y)."""
    pyautogui.doubleClick(x, y)
    print(f"Double-clicked ({x}, {y})")


def drag(x1, y1, x2, y2, duration=0.5):
    """Drag from (x1, y1) to (x2, y2)."""
    pyautogui.moveTo(x1, y1)
    pyautogui.drag(x2 - x1, y2 - y1, duration=duration)
    print(f"Dragged ({x1},{y1}) -> ({x2},{y2})")


def scroll(amount, x=None, y=None):
    """Scroll up (positive) or down (negative)."""
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(amount)
    print(f"Scrolled {amount}")


def get_position():
    """Print current mouse position."""
    pos = pyautogui.position()
    print(f"{pos.x},{pos.y}")
    return pos


def get_screen_size():
    """Print screen dimensions."""
    size = pyautogui.size()
    print(f"{size.width}x{size.height}")
    return size


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: mouse.py [click|rclick|move|pos|size|scroll] [args...]")
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "click":
        click(int(args[0]), int(args[1]))
    elif cmd == "rclick":
        right_click(int(args[0]), int(args[1]))
    elif cmd == "move":
        move(int(args[0]), int(args[1]))
    elif cmd == "pos":
        get_position()
    elif cmd == "size":
        get_screen_size()
    elif cmd == "scroll":
        amt = int(args[0])
        x = int(args[1]) if len(args) > 2 else None
        y = int(args[2]) if len(args) > 2 else None
        scroll(amt, x, y)
    elif cmd == "dclick":
        double_click(int(args[0]), int(args[1]))
    else:
        print(f"Unknown command: {cmd}")
