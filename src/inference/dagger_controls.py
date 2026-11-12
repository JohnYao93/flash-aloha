"""Task-independent optional DAgger pedal input; no robot or camera access."""
from __future__ import annotations
import select
import threading
from typing import Any, Callable

DEFAULT_DAGGER_FOOTPEDAL_DEVICE = (
    "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
)


DAGGER_FOOTPEDAL_KEY_CODES = {
    2: "1",
    3: "2",
    4: "3",
    79: "1",
    80: "2",
    81: "3",
}


def dagger_footpedal_key(event_type: int, event_code: int, event_value: int) -> str | None:
    """Translate one Linux input event into a single DAgger pedal press."""
    if event_type != 1 or event_value != 1:
        return None
    return DAGGER_FOOTPEDAL_KEY_CODES.get(int(event_code))


def open_optional_dagger_footpedal(reader: "DaggerFootpedalReader", *, logger=print) -> bool:
    """Use direct USB capture when available; browser control needs no device."""
    try:
        reader.open()
    except RuntimeError:
        reader.close()
        logger("[DAGGER] Using website buttons or keys 1/2/3; direct pedal input is unavailable.")
        return False
    return True


class DaggerFootpedalReader:
    """Read the dedicated USB footswitch independently of desktop focus."""

    def __init__(
        self,
        device_path: str,
        callback: Callable[[str], None],
        *,
        logger: Callable[[str], None] = print,
        input_device_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.device_path = str(device_path)
        self.callback = callback
        self.logger = logger
        self.input_device_factory = input_device_factory
        self.device: Any | None = None
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def open(self) -> None:
        if self.device is not None:
            return
        device = None
        try:
            factory = self.input_device_factory
            if factory is None:
                from evdev import InputDevice

                factory = InputDevice
            device = factory(self.device_path)
            # Exclusive ownership is intentional: otherwise the same physical
            # press also reaches whichever browser/terminal has focus and can
            # be processed twice or edit a browser field.
            device.grab()
        except Exception as exc:
            if device is not None:
                try:
                    device.close()
                except OSError:
                    pass
            raise RuntimeError(
                "Cannot exclusively open the DAgger footpedal at "
                f"{self.device_path!r}: {exc}. Check that the PCsensor pedal is "
                "connected and that this user can read the input device."
            ) from exc
        self.device = device
        self.logger(
            f"[DAGGER Pedal] direct input active: {self.device_path} "
            "(exclusive; browser/terminal focus is irrelevant)"
        )

    def start(self) -> None:
        if self.device is None:
            self.open()
        if self.thread is not None:
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="dagger-footpedal",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        assert self.device is not None
        try:
            while not self.stop_event.is_set():
                readable, _, _ = select.select([self.device.fd], [], [], 0.1)
                if not readable:
                    continue
                try:
                    events = self.device.read()
                except BlockingIOError:
                    continue
                for event in events:
                    key = dagger_footpedal_key(
                        event.type,
                        event.code,
                        event.value,
                    )
                    if key is not None:
                        self.logger(f"[DAGGER Pedal] pedal {key} pressed")
                        self.callback(key)
        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc
                self.logger(
                    f"[DAGGER Pedal] ERROR: direct input stopped: {exc}. "
                    "Use the hardware emergency stop if robot control is unsafe."
                )

    def close(self) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread is not None:
            thread.join(timeout=1.0)
            self.thread = None
        device = self.device
        self.device = None
        if device is not None:
            try:
                device.ungrab()
            except OSError:
                pass
            device.close()
