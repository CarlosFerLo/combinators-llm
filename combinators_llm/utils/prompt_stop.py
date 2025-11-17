def prompt_stop():
    """
    Check if the user has requested to stop training by entering 'stop' in the console.
    Returns True if training should be stopped, False otherwise.
    """
    import select
    import sys
    import termios
    import tty

    print("Press 's' then Enter to stop training early.")

    # Save the terminal settings
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        # Set terminal to raw mode
        tty.setraw(sys.stdin.fileno())

        # Check if there is input available
        dr, dw, de = select.select([sys.stdin], [], [], 0)
        if dr:
            char = sys.stdin.read(1)
            if char.lower() == "s":
                return True
    finally:
        # Restore terminal settings
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return False
