"""Keep this process's OpenCV window above other windows on X11 (and XWayland).

OpenCV's Qt backend ignores WND_PROP_TOPMOST, and GNOME stacks a window that
opens without input focus below the focused window. The replay overlay opens
while the terminal has focus (it reads the pedal keys), so after the first take
it appeared behind the terminal. This asks the window manager to keep it above
(_NET_WM_STATE_ABOVE) without moving input focus away from the terminal.
"""
import os

_STATE_ADD = 1
_SOURCE_APPLICATION = 1


def keep_window_above(title):
    """Mark this process's window named ``title`` keep-above; True once sent.

    False while the window is not mapped yet, or without X11/python-xlib.
    Never raises: a missing window manager hint must not stop the collector.
    """
    if not os.environ.get('DISPLAY'):
        return False
    try:
        from Xlib import X, display
        from Xlib.protocol import event
        connection = display.Display()
    except Exception:
        return False
    try:
        root = connection.screen().root
        atom = connection.intern_atom
        clients = root.get_full_property(atom('_NET_CLIENT_LIST'), X.AnyPropertyType)
        for window_id in (clients.value if clients else []):
            window = connection.create_resource_object('window', window_id)
            pid = window.get_full_property(atom('_NET_WM_PID'), X.AnyPropertyType)
            name = window.get_full_property(atom('_NET_WM_NAME'), X.AnyPropertyType)
            if pid is None or pid.value[0] != os.getpid() or name is None:
                continue
            value = name.value.decode('utf-8', 'replace') if isinstance(name.value, bytes) else name.value
            if value != title:
                continue
            message = event.ClientMessage(window=window, client_type=atom('_NET_WM_STATE'), data=(
                32, [_STATE_ADD, atom('_NET_WM_STATE_ABOVE'), 0, _SOURCE_APPLICATION, 0]))
            root.send_event(message, event_mask=X.SubstructureRedirectMask | X.SubstructureNotifyMask)
            connection.flush()
            return True
        return False
    except Exception:
        return False
    finally:
        connection.close()
